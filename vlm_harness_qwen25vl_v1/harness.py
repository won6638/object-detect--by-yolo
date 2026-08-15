
from __future__ import annotations
import argparse, hashlib, json, os, tempfile, time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import cv2
from jsonschema import Draft202012Validator
import requests

from provider import LlamaCppProvider

def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

def atomic_write_json(path: Path, obj: dict):
    payload = json.dumps(obj, ensure_ascii=False, indent=2)
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix=path.name+".", suffix=".tmp", text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(payload); f.flush(); os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.remove(temp)

def read_jsonl(path: Path):
    if not path.exists(): return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                obj = json.loads(line)
                if isinstance(obj, dict): rows.append(obj)
            except json.JSONDecodeError:
                pass
    return rows

def now_iso():
    return datetime.now().astimezone().isoformat(timespec="milliseconds")

def image_quality(path: Path, cfg: dict):
    if not path.exists(): raise FileNotFoundError(path)
    im = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if im is None: raise ValueError("image decode failed")
    h, w = im.shape[:2]
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    mean = float(gray.mean())
    lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    flags = []
    if min(h,w) < cfg.get("small_side_px",96): flags.append("small_crop")
    if lap < cfg.get("blur_laplacian_variance",60): flags.append("possibly_blurry")
    if mean < cfg.get("dark_mean",25): flags.append("very_dark")
    if mean > cfg.get("bright_mean",235): flags.append("very_bright")
    return {"width":w,"height":h,"mean_brightness":round(mean,2),
            "laplacian_variance":round(lap,2),"quality_flags":flags}

def validate_and_clean(data: dict, schema: dict):
    errors = list(Draft202012Validator(schema).iter_errors(data))
    if errors:
        return data, False, "UNCERTAIN", True, [e.message for e in errors]

    c = deepcopy(data); warnings=[]; escalation=False
    if c["recognition_status"] == "unknown":
        for k in ("object_name","category","brand","product_name","model"):
            c[k] = None
        c["brand_evidence"]=c["product_evidence"]=c["model_evidence"]="none"
        return c, True, "UNCERTAIN", False, warnings

    if c.get("brand") is not None and c.get("brand_evidence") == "none":
        c["brand"] = None; warnings.append("brand removed: no visible evidence"); escalation=True
    if c.get("product_name") is not None and c.get("product_evidence") == "none":
        c["product_name"] = None; warnings.append("product_name removed: no visible evidence"); escalation=True
    if c.get("model") is not None and c.get("model_evidence") == "none":
        c["model"] = None; warnings.append("model removed: no visible evidence"); escalation=True

    if c["recognition_status"] == "identified" and not c.get("object_name"):
        c["recognition_status"] = "uncertain"
        warnings.append("identified downgraded: no object_name")
        escalation=True

    decision = "ACCEPT" if c["recognition_status"] == "identified" and not escalation else "UNCERTAIN"
    return c, True, decision, escalation, warnings

def hash_file(path: Path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024), b""): h.update(chunk)
    return h.hexdigest()

class Worker:
    def __init__(self, cfg, schema):
        self.cfg=cfg; self.schema=schema
        self.root=Path(cfg["capture_root"]).resolve()
        self.queue=self.root/cfg["queue_file"]
        self.provider=LlamaCppProvider(cfg["provider"], schema)
        self.cache_path=self.root/"harness_cache.json"
        self.cache=read_json(self.cache_path) if self.cache_path.exists() else {}

    def pending(self):
        out=[]
        for row in read_jsonl(self.queue):
            p=self.root/Path(row["record"])
            if p.exists():
                rec=read_json(p)
                if rec.get("llm",{}).get("status")=="pending":
                    out.append((row,p,rec))

        if self.cfg.get("coalesce_same_object",True):
            newest={}
            superseded=[]
            for item in out:
                oid=item[2].get("object_id")
                prev=newest.get(oid)
                if prev is None or item[2].get("timestamp","") >= prev[2].get("timestamp",""):
                    if prev: superseded.append(prev)
                    newest[oid]=item
                else: superseded.append(item)
            for _,p,r in superseded:
                r["llm"]["status"]="cancelled"
                r["llm"].setdefault("harness",{})["cancel_reason"]="superseded_by_newer_capture"
                atomic_write_json(p,r)
            out=list(newest.values())

        out.sort(key=lambda x:x[2].get("timestamp",""), reverse=self.cfg.get("latest_first",True))
        return out

    def process_one(self):
        items=self.pending()
        if not items: return False
        _, path, rec=items[0]
        start=time.perf_counter()
        llm=rec.setdefault("llm",{})
        h=llm.setdefault("harness",{})
        llm["status"]="processing"
        h.update({"provider":"llamacpp","model":self.cfg["provider"]["model"],
                  "processing_at":now_iso(),"schema_valid":None,"decision":None})
        atomic_write_json(path,rec)

        try:
            img=self.root/Path(rec["image"]["llm"]["file"])
            qt=time.perf_counter(); quality=image_quality(img,self.cfg["quality"])
            image_ms=(time.perf_counter()-qt)*1000

            detector=rec.get("detector")
            hint=None if not detector else {
                "class_name":detector.get("label_hint"),
                "confidence":detector.get("confidence")
            }

            cache_key=hashlib.sha256(
                (hash_file(img)+"|"+self.cfg["provider"]["model"]+"|v1").encode()
            ).hexdigest()

            if cache_key in self.cache:
                raw=self.cache[cache_key]["data"]; provider_ms=0.0
                meta={"cache":True}; cache_hit=True
            else:
                raw,provider_ms,meta=self.provider.identify(
                    image_path=img,
                    selection_method=rec["selection"]["method"],
                    yolo_hint=hint,
                    quality_flags=quality["quality_flags"]
                )
                self.cache[cache_key]={"data":raw,"cached_at":now_iso()}
                atomic_write_json(self.cache_path,self.cache)
                cache_hit=False

            vt=time.perf_counter()
            cleaned,valid,decision,escalate,warnings=validate_and_clean(raw,self.schema)
            validation_ms=(time.perf_counter()-vt)*1000

            if cleaned.get("recognition_status")=="unknown": status="unknown"
            elif decision=="ACCEPT": status="completed"
            else: status="uncertain"

            total_ms=(time.perf_counter()-start)*1000
            llm["status"]=status; llm["response"]=cleaned
            h.update({
                "schema_valid":valid,"decision":decision,
                "escalation_recommended":escalate,"warnings":warnings,
                "quality":quality,"cache_hit":cache_hit,"raw_response":raw,
                "provider_metadata":meta,
                "latency":{
                    "image_prepare_ms":round(image_ms,2),
                    "provider_latency_ms":round(provider_ms,2),
                    "validation_ms":round(validation_ms,2),
                    "total_latency_ms":round(total_ms,2)
                },"finished_at":now_iso()
            })
            atomic_write_json(path,rec)
            print(f"[done] {rec['request_id']} status={status} decision={decision} provider={provider_ms:.0f}ms total={total_ms:.0f}ms")
        except requests.Timeout as e:
            llm["status"]="timeout"; h["error"]={"type":"provider_timeout","message":str(e)}
            h["finished_at"]=now_iso(); atomic_write_json(path,rec)
            print("[timeout]",rec.get("request_id"))
        except Exception as e:
            llm["status"]="failed"; h["error"]={"type":type(e).__name__,"message":str(e)}
            h["finished_at"]=now_iso(); atomic_write_json(path,rec)
            print("[failed]",rec.get("request_id"),type(e).__name__,e)
        return True

    def watch(self):
        print("[worker] queue:",self.queue)
        try:
            while True:
                if not self.process_one():
                    time.sleep(max(.05,self.cfg.get("poll_interval_ms",100)/1000))
        except KeyboardInterrupt:
            print("\n[worker] stopped")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",default="harness_config.json")
    ap.add_argument("--health",action="store_true")
    ap.add_argument("--once",action="store_true")
    a=ap.parse_args()
    cfg=read_json(Path(a.config))
    schema=read_json(Path("vision_response.schema.json"))
    w=Worker(cfg,schema)
    if a.health:
        print(json.dumps(w.provider.health(),ensure_ascii=False,indent=2)); return
    if a.once:
        if not w.process_one(): print("[worker] no pending requests")
        return
    w.watch()

if __name__=="__main__": main()
