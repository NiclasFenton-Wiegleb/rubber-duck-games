import inspect
import huggingface_hub as h

out = []
for n in ["download_bucket_files", "sync_bucket", "list_bucket_tree",
          "get_bucket_paths_info", "bucket_info"]:
    fn = getattr(h, n)
    try:
        sig = str(inspect.signature(fn))
    except (ValueError, TypeError):
        sig = "(<no signature>)"
    doc = (inspect.getdoc(fn) or "")[:1200]
    out.append(f"### {n}{sig}\n{doc}")

with open("hf_sig.txt", "w", encoding="utf-8") as f:
    f.write("\n\n".join(out))
print("done")
