"""Annex B NAL parsing, RTP/pcap build/parse, SDP generation for H264/H265."""
import base64
import random
import time

from scapy.all import TCP, UDP, Raw, rdpcap


def base64_nalu(nalu):
    return base64.b64encode(nalu).decode('ascii')

def generate_sdp(codec, port=5004, fps=30, vps=None, sps=None, pps=None):
    if codec == 'h264':
        sdp = (
            "v=0\r\n"
            "o=- 0 0 IN IP4 10.0.0.2\r\n"
            "s=H264 Test Stream\r\n"
            "t=0 0\r\n"
            f"m=video {port} RTP/AVP 96\r\n"
            "a=rtpmap:96 H264/90000\r\n"
            f"a=framerate:{fps}\r\n"
            "a=control:trackID=1\r\n"
        )
    elif codec == 'h265':
        sdp = (
            "v=0\r\n"
            "o=- 0 0 IN IP4 10.0.0.2\r\n"
            "s=H265 Test Stream\r\n"
            "t=0 0\r\n"
            f"m=video {port} RTP/AVP 96\r\n"
            "a=rtpmap:96 H265/90000\r\n"
        )
        # Optional fmtp with base64 VPS/SPS/PPS
        if vps and sps and pps:
            sdp += (
                "a=fmtp:96 "
                f"sprop-vps={base64_nalu(vps)}; "
                f"sprop-sps={base64_nalu(sps)}; "
                f"sprop-pps={base64_nalu(pps)}\r\n"
            )
        sdp += "a=control:trackID=1\r\n"
    else:
        raise ValueError("Unsupported codec")
    return sdp

def parse_annexb(file_bytes):
    """Extract NAL units from Annex B formatted byte stream."""
    nalus = []
    i = 0
    while i < len(file_bytes) - 4:
        if file_bytes[i:i+3] == b'\x00\x00\x01':
            start = i + 3
        elif file_bytes[i:i+4] == b'\x00\x00\x00\x01':
            start = i + 4
        else:
            i += 1
            continue
        end = start
        while end < len(file_bytes) - 4:
            if file_bytes[end:end+3] == b'\x00\x00\x01' or file_bytes[end:end+4] == b'\x00\x00\x00\x01':
                break
            end += 1
        nalus.append(file_bytes[start:end])
        i = end
    return nalus

def build_rtp_packet(payload, seq, timestamp, marker, ssrc=0x12345678, pt=96, capture_time=None):
    import struct
    from scapy.all import Ether, IP, UDP, Raw
    v_p_x_cc = 0x80  # V=2, P=0, X=0, CC=0
    m_pt = (0x80 if marker else 0x00) | (pt & 0x7F)  # M | PT
    rtp_header = struct.pack("!BBHII", v_p_x_cc, m_pt, seq & 0xFFFF, timestamp & 0xFFFFFFFF, ssrc & 0xFFFFFFFF)
    pkt = (
        Ether(src="00:11:22:33:44:55", dst="66:77:88:99:AA:BB") /
        IP(src="10.0.0.1", dst="10.0.0.2") /
        UDP(sport=5004, dport=5004) /
        Raw(load=rtp_header + payload)
    )
    if capture_time is None:
        capture_time=timestamp/90000.0
    pkt.time = capture_time
    return pkt

def packetize_h264(nalu, seq, timestamp, max_payload_size, capture_time=None, ssrc=0x12345678):
    """Returns list of RTP packets for a single H.264 NALU."""
    packets = []
    nalu_type = nalu[0] & 0x1F

    #print("NUT",nalu_type)

    if len(nalu) <= max_payload_size:
        packets.append((build_rtp_packet(nalu, seq, timestamp, marker=1, capture_time=capture_time, ssrc=ssrc), seq + 1))
    else:
        fu_indicator = (nalu[0] & 0xE0) | 28  # FU-A type is 28
        fu_header = (1 << 7) | nalu_type      # Start bit
        offset = 1
        while offset < len(nalu):
            remaining = len(nalu) - offset
            size = min(remaining, max_payload_size - 2)
            fu_payload = bytes([fu_indicator, fu_header]) + nalu[offset:offset+size]
            marker = 1 if offset + size == len(nalu) else 0
            packets.append((build_rtp_packet(fu_payload, seq, timestamp, marker, capture_time=capture_time, ssrc=ssrc), seq + 1))
            seq += 1
            fu_header = nalu_type  # Clear start bit, no end yet
            offset += size
    return packets

def packetize_h265(nalu: bytes,
                   seq: int,
                   timestamp: int,
                   max_payload_size: int,
                   capture_time=None, ssrc=0x12345678):
    """
    Returns a list of (rtp_packet_bytes, next_seq) for a single H.265 NALU.
    Each RTP packet is either:
      - one single NALU if it fits, or
      - a series of FU (fragmentation unit) packets, each carrying:
        ][payload-bytes… up to max_payload_size]
    """
    packets = []

    # Case A: If the entire NALU fits into one RTP payload, send it “raw.”
    if len(nalu) <= max_payload_size:
        rtp     = build_rtp_packet(nalu,
                                   seq,
                                   timestamp,
                                   marker=1,
                                   capture_time=capture_time,
                                   ssrc=ssrc)
        packets.append((rtp, seq + 1))
        return packets

    # Otherwise we fragment: the first two bytes are the original header
    H0 = nalu[0]   # first NAL header byte
    H1 = nalu[1]   # second NAL header byte
    # Extract the 6-bit nal_unit_type from H0:
    nalu_type = (H0 >> 1) & 0x3F

    # FU Indicator is *two bytes*:
    #   byte0 = F | (49<<1) | layer_id_msb
    #   byte1 = layer_id_lsb<<3 | tid_plus1
    fu_ind0 = (H0     & 0x81)       | (49 << 1)
    fu_ind1 = (H1     & 0xFF)       # copy all of original H1 unchanged

    # FU Header’s “base” (bits7..0) = [S/E=0][RSV=0][orig_type (6 bits)].
    # We do NOT include any part of H1 here.  The 6 bits of “orig_type”
    # go into FU_HDR bits4..0.  That means:
    #
    #   FU_HDR:  1 byte:
    #     bit7 = S, bit6 = E, bit5 = 0, bits4..0 = nalu_type.
    fu_header_base = (nalu_type & 0x3F)  # (nalu_type<<0),
                                          # we will OR in S/E bits at 0x80/0x40.

    # Now walk through the rest of the NALU (after those two header bytes).
    offset = 2
    first  = True

    # Each FU packet must send:
    #   [1 byte FU_IND][1 byte FU_HDR][1 byte original_H1][<payload>]
    #
    # That is 3 bytes of "overhead" before actual payload data.
    # So if max_payload_size = N, we can only put up to (N−3) bytes of payload in each FU.
    payload_capacity = max_payload_size - 3

    while offset < len(nalu):
        remaining = len(nalu) - offset
        chunk_size = min(remaining, payload_capacity)

        # Start with the “base” fu_header (just orig_type in bits4..0):
        fu_header = fu_header_base
        if first:
            fu_header |= 0x80  # set “S” (start)
        if offset + chunk_size == len(nalu):
            fu_header |= 0x40  # set “E” (end)

        # Build the actual FU packet’s RTP payload:
        #  1 byte: fu_indicator
        #  1 byte: fu_header
        #  1 byte: original H[1]
        #  chunk_size bytes: raw NALU[offset:offset+chunk_size]
        fragment_payload = bytes([fu_ind0, fu_ind1, fu_header]) \
                      + nalu[offset : offset + chunk_size]

        marker = 1 if (fu_header & 0x40) else 0  # marker=1 on the very last fragment of this NALU
        rtp_pkt = build_rtp_packet(fragment_payload,
                                   seq,
                                   timestamp,
                                   marker=marker,
                                   capture_time=capture_time,
                                   ssrc=ssrc)
        packets.append((rtp_pkt, seq + 1))

        seq += 1
        offset += chunk_size
        first = False

    return packets

def annexb_to_pcap(h26x, codec='h264', fps=30.0, output="output.pcap", max_payload_size=1200):
    #from scapy.all import *
    from scapy.all import Ether, IP, TCP, UDP, Raw, wrpcap

    with open(h26x, 'rb') as f:
        data = f.read()

    nalus = parse_annexb(data)
    print(f"Found {len(nalus)} NAL units.")

    ts_increment = int(90000 / fps)
    timestamp = random.randint(0, 2**32 - 1)
    ssrc = random.randint(0, 2**32 - 1)
    seq = random.randint(0,65535)
    packets = []
    capture_time=time.time() # float unix time seconds

    # Construct RTSP DESCRIBE response with SDP
    sdp = generate_sdp(codec, fps)

    rtsp_response = (
        "RTSP/1.0 200 OK\r\n"
        "CSeq: 1\r\n"
        "Content-Type: application/sdp\r\n"
        f"Content-Length: {len(sdp)}\r\n"
        "\r\n" + sdp
    )

    tcp_payload = rtsp_response.encode("utf-8")

    pkt = (
        Ether(src="00:11:22:33:44:55", dst="66:77:88:99:AA:BB") /
        IP(src="10.0.0.2", dst="10.0.0.1") /
        TCP(sport=554, dport=50000, seq=1, ack=1, flags="PA") /
        Raw(load=tcp_payload)
    )

    pkt.time = capture_time
    packets.extend(pkt)

    for nalu in nalus:
        if codec == 'h264':
            rtp_packets = packetize_h264(nalu, seq, timestamp, max_payload_size, capture_time=capture_time, ssrc=ssrc)
            nalu_type = nalu[0] & 0x1F
            is_video = 1 <= nalu_type <= 5
        else:
            rtp_packets = packetize_h265(nalu, seq, timestamp, max_payload_size, capture_time=capture_time, ssrc=ssrc)
            nalu_type = (nalu[0] >> 1) & 0x3F
            is_video = 0 <= nalu_type <= 31

        packets.extend(pkt for pkt, _ in rtp_packets)
        seq = rtp_packets[-1][1]
        if is_video:
            capture_time+=(1.0/fps)
            timestamp += ts_increment  # advance timestamp per NALU/frame

    wrpcap(output, packets)
    print(f"Wrote {len(packets)} RTP packets to {output}")

def parse_pcap(filename,
                 sdp_tcp_port=554,
                 rtp_ports=None):
    """
    Parse a pcapng file containing one SDP over TCP and RTP packets.

    Args:
        filename (str): Path to the pcapng file.
        sdp_tcp_port (int): TCP port where the SDP response is carried (default 554).
        rtp_ports (set[int] or None): UDP ports to consider for RTP (default any UDP packet).
        rtp_payload_offset (int): Number of bytes in the RTP header to skip (default 12).

    Returns:
        tuple:
            sdp_text (str): The SDP description as a UTF-8 string.
            rtp_payloads (list[bytes]): List of raw RTP payloads (header stripped).
            rtp_times (list[float]): List of packet capture timestamps (pkt.time) for each RTP payload.
    """
    packets = rdpcap(filename)
    sdp_text = None
    rtp_payloads = []
    rtp_times = []

    for pkt in packets:
        # Extract SDP over TCP
        if TCP in pkt and Raw in pkt:
            tcp = pkt[TCP]
            if tcp.sport == sdp_tcp_port or tcp.dport == sdp_tcp_port:
                data = pkt[Raw].load
                if data.startswith(b"RTSP/1.0 200 OK") and b"Content-Type: application/sdp" in data:
                    header, _, body = data.partition(b"\r\n\r\n")
                    try:
                        sdp_text = body.decode('utf-8', errors='replace')
                    except Exception:
                        sdp_text = body.decode('latin-1', errors='replace')
                    # Continue to collect RTP packets
                    continue
        # Extract RTP from UDP
        if UDP in pkt and Raw in pkt:
            udp = pkt[UDP]
            if rtp_ports is None or udp.sport in rtp_ports or udp.dport in rtp_ports:
                payload = pkt[Raw].load
                rtp_payloads.append(payload)
                # record capture timestamp
                rtp_times.append(float(pkt.time))

    return sdp_text, rtp_payloads, rtp_times

class pcap_packet_streamer:
    def __init__(self, pcap_file):
        self.sdp, self.rtp_packets, self.rtp_times=parse_pcap(pcap_file)
        self.rtp_times=[c-self.rtp_times[0] for c in self.rtp_times]
        self.time=0
        self.n=0

    def get_sdp(self):
        return self.sdp

    def duration(self):
        return self.rtp_times[-1]

    def get_packets(self, time_delta):
        self.time+=time_delta
        ret=[]
        while True:
            if self.n>=len(self.rtp_times):
                break
            t=self.rtp_times[self.n]
            if t>self.time:
                break
            ret.append(self.rtp_packets[self.n])
            self.n+=1
        return ret