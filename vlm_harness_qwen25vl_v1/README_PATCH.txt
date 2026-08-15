VLM Harness v1.2 policy patch
=============================

Change
------
- object_name/category are intentionally less conservative.
- General object type may be inferred from visible shape, parts and apparent function.
- A broad useful category is preferred over immediately returning unknown.
- Brand/product_name/exact model remain strictly evidence-gated.
- YOLOE hint remains hint-only.

Replace
-------
provider.py

in:
C:\Users\who66\Desktop\yolo26_webcam\vlm_harness_qwen25vl_v1\

No additional accuracy benchmark is required before proceeding.

Next
----
1. Keep llama-server running.
2. Make sure harness_config.json has:
   "capture_root": "../capture"
3. Start the long-running worker:
   python .\harness.py
4. In another terminal run the YOLOE desktop pipeline.
5. Select object -> C.
6. Harness automatically consumes pending queue entries and updates record.json.

Future model replacement boundary
---------------------------------
The following remain unchanged:
- Vision front-end
- capture/ layout
- record.json
- queue.jsonl
- validator/policy layer
- UI contract

Only the Provider adapter/config changes for a future cloud VLM.
