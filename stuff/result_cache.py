"""Pickle-backed result list with add/get/delete by key match."""
import os
import pickle

import stuff
from datetime import datetime

class ResultCache:

    def __init__(self, resultfile):
        self.resultfile=resultfile
        self.results=[]
        if os.path.isfile(self.resultfile):
            with open(self.resultfile, 'rb') as handle:
                self.results = pickle.load(handle)
            print(f"Loaded {len(self.results)} cached results from {self.resultfile}")

    def save(self):
        stuff.save_atomic_pickle(self.results, self.resultfile)

    def add(self, result):
        result["add_time"]=datetime.now()
        self.results.append(result)
        self.save()

    def __find__(self, match):
        ret=[]
        for i,r in enumerate(self.results):
            ok=True
            for k in match:
                if k not in r or r[k]!=match[k]:
                    ok=False
            if ok:
                ret.append(i)
        return ret

    def get(self, match):
        idx=self.__find__(match)
        if len(idx)==0:
            return None
        if len(idx)==1:
            return self.results[idx[0]]
        time_now=datetime.now()
        # if multiple matches, return the most recently added
        ages=[ (time_now - self.results[i]["add_time"]).total_seconds() for i in idx ]
        index_min=min(range(len(ages)), key=ages.__getitem__)
        return self.results[idx[index_min]]

    def delete(self, match):
        idx=self.__find__(match)
        if len(idx)==0:
            return 0
        for i in sorted(idx, reverse=True):
            del self.results[i]
        self.save()
        return len(idx)