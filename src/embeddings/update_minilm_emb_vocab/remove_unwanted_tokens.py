# src/embeddings/update_minilm_emb_vocab/remove_unwanted_tokens.py
from pathlib import Path
from transformers import AutoTokenizer

from src.utils.helpers import load_config

ROOT = Path(__file__).resolve().parents[3]

config = load_config()
MODEL_NAME = config["retrieval"]["model"]
BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOKENIZER_DIR = ROOT / "models" / "kubernetes_upd_tokenizer_vocab"

TOKENS_TO_REMOVE = {
    "examplecontainerid1234567890abcdef",
    "hostnames-632524106-bbpiw",
    "hostnames-632524106-tlaok",
    "hostnames-632524106-ly40y",
    "nginx-deployment-c4747d96c",
    "nginx-deployment-75675f5897",
    "kube-proxy-kubernetes-node-i4c4",
    "node-debugger-mynode-pdx84",
    "KUBE-SEP-WNBA2IHDGP2BOBGZ",
    "KUBE-SEP-57KPRZ3JQVENLNBR",
    "eipalloc-def1234abc4567890",
    "fluentd-cloud-logging-kubernetes-node-i4c4",
    "dra-example-driver-kubeletplugin-4sk2x",
    "dra-example-driver-kubeletplugin-cttr2",
    "gke-main-default-pool-dca1511c-d17b",
    "patch-demo-1307768864-c86dc",
    "patch-demo-1307768864-69308",
    "nginx-701339712-e0qfq",
    "userinfo-v1-authentication-k8s-io",
    "kubernetes-downwardapi-volume-example-2",
    "k8s-apparmor-example-deny-write",
    "aws-load-balancer-private-ipv4-addresses",
    "aws-load-balancer-access-log-s3-bucket-name",
    "aws-load-balancer-access-log-s3-bucket-prefix",
    "key1",
    "nginx-deployment-7768647bf9-rkxlw",
    "example-secret-2-t52t6g96d8",
    "mysql-ddc65b868-glc5m",
    "c0izixklcxtmnze7",
    "database-creds-5hdh7hhgfk",
    "KUBE-SVC-NWV5X2332I4OT4T3",
    "nginx-deployment-3066724191-08mng",
    "nginx-deployment-3066724191",
    "pvc-b95650f8-b7b5-11e6-9d58-0ed433a7dd94",
    "mongo-75f59d57f4",
    "kubernetes-node-2g2d",
    "kubernetes-bootcamp-644c5687f4-4hjvf",
    "redis-follower-dddfbdcc9-qrt5k",
    "immutable-configmap-volume-5fdb88fcc8-29v8n",
    "frontend-85595f5bf9-l2l54",
    "hello-world-cdd4458f4-m47c8",
    "nginx-deployment-75675f5897-qqcnn",
    "gke-cluster-1-default-pool-e0b8d269-cpuc",
    "kubernetes-node-9l2t",
    "hello-world-2895499144-2e5uh",
    "pvc-ba3bfe9c-413e-4b95-a2c0-3ea8a54dbab4",
    "pvc-8cd0df54-4044-11e9-b2bb-42010a800002",
    "gke-main-default-pool-dca1511c-d17b",
    "frontend-85595f5bf9-5df5m",
    "rollout-demo-canary-8e5c0d9f6a-xyz78",
    "nginx-deployment-595696685f",
    "hello-world-2895499144-bsbk5",
    "coredns-5644d7b6d9-mh9ll",
    "e2e-test-node-pool-4lw4",
    "nginx-deployment-148880595-6zgi1",
    "pvc-0c04d7f0-787a-4977-8da3-d9d3a6d8d752",
    "pod-configmap-envFrom",
    "nginx-701339712-e0qfq",
    "coredns-3297075139-61lj3",
    "frontend-85595f5bf9-5tqhb",
    "my-nginx-3800858182-kna2y",
    "hello-world-2895499144-9m4h1",
    "nginx-deployment-7768647bf9-b4b9s",
    "frontend-85595f5bf9-qbzwm",
    "nginx-deployment-1564180365-70iae",
    "cattle-2263376956-41xy6",
    "nginx-deployment-1771418926",
    "pvc-b2c73489-e70b-4a4e-9ec1-9eab439aa43e",
    "game-config-5-m67dt67794",
    "cattle-2263376956-p5p3i",
    "nginx-deployment-7fb96c846b-mlgns",
    "configmap-env-var-59cfc64f7d-dpr98",
    "ec2-10-100-0-1",
    "cattle-2263376956-kw466",
    "constraints-mem-demo-4-ctr",
    "pvc-2bf00408-d366-4a12-bad0-1869c65d0bee",
    "nginx-65899c769f-wv2gp",
    "nginx-deployment-1564180365-hysrc",
    "wordpress-mysql-1894417608-x5dzt",
    "com-cluster-1-node-2-fghij",
    "job-backoff-limit-per-index-failindex-0-4g4cm",
    "nginx-deployment-7fb96c846b-22567",
    "coredns-5644d7b6d9-pqd2t",
    "hello-world-2895499144-9m4h1",
    "kubernetes-bootcamp-644c5687f4-wp67j",
    "hello-world-2895499144-segjf",
    "job-pod-failure-policy-failjob-8ckj8",
    "gke-test-default-pool-239f5d02-gyn2",
    "kubernetes-node-tf0f",
    "coredns-7bb9c7b568-hsptc",
    "configmap-sidecar-container-5fb59f558b-wnmgk",
    "web-1",
    "web-0",
    "nginx-deployment-75675f5897-7ci7o",
    "hello-world-2895499144-9m4h1",
    "frontend-85595f5bf9-zchwc",
    "my-nginx-2035384211-u2c7e",
    "cattle-2263376956-n4v97",
    "snowflake-3968820950-vgc4n",
    "job-backoff-limit-per-index-failindex-1-2bgdj",
    "example-configmap-2-g2hdhfc6tk",
    "nginx-deployment-1006230814-6winp",
    "nginx-deployment-1564180365-khku8",
    "immutable-configmap-volume-5fdb88fcc8-n5jx4",
    "immutable-configmap-volume-78b6fbff95-5gsfh",
    "local-path-provisioner-d6d9f7ffc-lw9lh",
    "quota-mem-cpu-demo-2-ctr",
    "curl-131556218-9fnch",
    "kubernetes-node-pb41",
    "job-pod-failure-policy-config-issue-k6pvp",
    "nginx-deployment-148880595-4zdqq",
    "snowflake-3968820950-9dgr8",
    "my-nginx-3800858182-e9ihh",
    "slf4j-log4j12-1",
    "foo-sidecar-19700101-000000",
    "redis-leader-fb76b4755-xjr2n",
    "nginx-app-5jyvm",
    "job-backoff-limit-per-index-failindex-2-vs6lt",
    "nginx-deployment-1989198191",
    "example-configmap-1-42cfbf598f",
    "nginx-deployment-2035384211",
    "nginx-deployment-1564180365",
    "hello-node-5f76cf6ccf-br9b5",
    "hello-world-2895499144-1jaz9",
    "nginx-deployment-1771418926-7o5ns",
    "frontend-85595f5bf9-l9c8z",
    "pvc-8cbd7b2e-4044-11e9-b2bb-42010a800002",
    "job-wq-1-xe5hj",
    "immutable-configmap-volume-78b6fbff95-7vcj4",
    "hello-world-2895499144-m1pwt",
    "frontend-85595f5bf9-7zmg5",
    "redis-follower-dddfbdcc9-82sfr",
    "hostnames-632524106-bbpiw",
    "nginx-deployment-1771418926-r18az",
    "frontend-85595f5bf9-cpskg",
    "web-2",
    "nginx-deployment-618515232",
    "patch-demo-1307768864-69308",
    "mysql-pass-c57bb4t7mf",
    "qos-demo-4-ctr-1",
    "configmap-volume-6b976dfdcf-tbc6r",
    "nginx-deployment-148880595-rwovn",
    "nginx-deployment-1370807587-fz9sd",
    "d3e48dbf-40da-47c3-a7b9-f7d54d1051c3-gpu-0",
    "kube-proxy-m6hwf",
    "zk-0",
    "zk-1",
    "hello-node-ccf4b9788-4jn97",
    "guestbook-redis-replica-2q2yf",
    "q1bKLFOyUiosTfawzA93TzZIDzH2TNa2SMm0zEiPKTUwME6BkEo6Sql5yUWVBSWpKUGphaWpxSVAfekBOZbBhaEW",
    "byAubWQgZmlsZV1cblxuICAgIEEgLS0-IEJcbiAgICBBIC0tPiBDXG4gICAgQSAtLT4gRFxuXG4gICAgY2xhc3NEZWYgYm94IGZpbGw6I2ZmZixzdHJva2U6IzAwMCxzdHJva2Utd2lkdGg6MXB4LGNvbG9yOiMwMDA7XG4gICAgY2xhc3MgQSxCLEMsRCBib3giLCJtZXJtYWlkIjoie1xuICBcInRoZW1lXCI6IFwiZGVmYXVsdFwiXG59IiwidXBkYXRlRWRpdG9yIjpmYWxzZSwiYXV0b1N5bmMiOnRydWUsInVwZGF0ZURpYWdyYW0iOmZhbHNlfQ",
    "YWdlcy8gZm9sZGVyXVxuICAgIHN1YmdyYXBoIHdbIF1cbiAgICBkaXJlY3Rpb24gVEJcbiAgICBEWzQuIFVzZSBmaWd1cmUgc2hvcnRjb2RlPGJyPnRvIHJlZmVyZW5jZSBzdmcgb3I8YnI-cG5nIGZpbGUgaW48YnI-cGFnZSAubWQgZmlsZV0gLS0-XG4gICAgRVs1LiBBZGQgY2FwdGlvbl1cbiAgICBlbmRcbiAgICBBIC0tPiBCXG4gICAgQiAtLT4gQ1xuICAgIEMgLS0-IHdcbiAgICBjbGFzc0RlZiBib3ggZmlsbDojZmZmLHN0cm9rZTojMDAwLHN0cm9rZS13aWR0aDoxcHgsY29sb3I6IzAwMDtcbiAgICBjbGFzcyBBLEIsQyxELEUsdyBib3hcbiAgICAiLCJtZXJtYWlkIjoie1xuICBcInRoZW1lXCI6IFwiZGVmYXVsdFwiXG59IiwidXBkYXRlRWRpdG9yIjpmYWxzZSwiYXV0",
    "eyJjb2RlIjoiZmxvd2NoYXJ0IExRXG4gICAgQVsxLiBVc2UgbGl2ZSBlZGl0b3I8YnI-IHRvIGNyZWF0ZS9lZGl0PGJyPmRpYWdyYW1dXG4gICAgQlsyLiBTdG9yZSBkaWFncmFtPGJyPlVSTCBzb21ld2hlcmVdXG4gICAgQ1szLiBHZW5lcmF0ZSAuc3ZnIGZpbGU8YnI-YW5kIGRvd25sb2FkIHRvPGJyPmltYWdlcy8gZm9sZGVyXVxuICAgIHN1YmdyYXBoIHdbIF1cbiAgICBkaXJlY3Rpb24gVEJcbiAgICBEWzQuIFVzZSBmaWd1cmUgc2hvcnRjb2RlPGJyPnRvIHJlZmVyZW5jZSBzdmcgb3I8YnI-cG5nIGZpbGUgaW48YnI-cGFnZSAubWQgZmlsZV0gLS0-XG4gICAgRVs1LiBBZGQgY2FwdGlvbl1cbiAgICBlbmRcbiAgICBBIC0tPiBCXG4gICAgQiAtLT4gQ1xuICAgIEMgLS0-IHdcbiAgICBjbGFzc0RlZiBib3ggZmlsbDojZmZmLHN0cm9rZTojMDAwLHN0cm9rZS13aWR0aDoxcHgsY29sb3I6IzAwMDtcbiAgICBjbGFzcyBBLEIsQyxELEUsdyBib3hcbiAgICAiLCJtZXJtYWlkIjoie1xuICBcInRoZW1lXCI6IFwiZGVmYXVsdFwiXG59IiwidXBkYXRlRWRpdG9yIjpmYWxzZSwidXBkYXRlRGlhZ3JhbSI6ZmFsc2V9",
    "eyJjb2RlIjoiZ3JhcGggTFJcbiAgICBBWzEuIFVzZSBsaXZlIGVkaXRvcjxicj4gdG8gY3JlYXRlL2VkaXQ8YnI-ZGlhZ3JhbV0gLS0-XG4gICAgQlsyLiBTdG9yZSBkaWFncmFtPGJyPlVSTCBzb21ld2hlcmVdIC0tPlxuICAgIENbMy4gQ29weSBNZXJtYWlkIGNvZGU8YnI-dG8gcGFnZSBtYXJrZG93biBmaWxlXSAtLT5cbiAgICBEWzQuIEFkZCBjYXB0aW9uXVxuIFxuXG4gICAgY2xhc3NEZWYgYm94IGZpbGw6I2ZmZixzdHJva2U6IzAwMCxzdHJva2Utd2lkdGg6MXB4LGNvbG9yOiMwMDA7XG4gICAgY2xhc3MgQSxCLEMsRCBib3hcbiAgICAiLCJtZXJtYWlkIjoie1xuICBcInRoZW1lXCI6",
    "eyJjb2RlIjoiZmxvd2NoYXJ0XG4gICAgQVtEaWFncmFtPGJyPjxicj5JbmxpbmUgTWVybWFpZCBvcjxicj5TVkcgaW1hZ2UgZmlsZXNdXG4gICAgQltEaWFncmFtIENhcHRpb248YnI-PGJyPkFkZCBGaWd1cmUgTnVtYmVyLiBhbmQ8YnI-Q2FwdGlvbiBUZXh0XVxuICAgIENbRGlhZ3JhbSBSZWZlcnJhbDxicj48YnI-UmVmZXJlbmVuY2UgRmlndXJlIE51bWJlcjxicj5pbiB0ZXh0XVxuXG4gICAgY2xhc3NEZWYgYm94IGZpbGw6I2ZmZixzdHJva2U6IzAwMCxzdHJva2Utd2lkdGg6MXB4LGNvbG9yOiMwMDA7XG4gICAgY2xhc3MgQSxCLEMgYm94IiwibWVybWFpZCI6IntcbiAgXCJ0aGVtZVwiO",
    "aWd1cmUgc2hvcnRjb2RlPGJyPnRvIHJlZmVyZW5jZSAuc3ZnPGJyPmZpbGUgaW4gcGFnZTxicj4ubWQgZmlsZV0gLS0-XG4gICAgRVs1LiBBZGQgY2FwdGlvbl1cbiAgICBlbmRcbkEgLS0-IEJcbkIgLS0-IENcbkMgLS0-IHdcblxuICAgIGNsYXNzRGVmIGJveCBmaWxsOiNmZmYsc3Ryb2tlOiMwMDAsc3Ryb2tlLXdpZHRoOjFweCxjb2xvcjojMDAwO1xuICAgIGNsYXNzIEEsQixDLEQsRSx3IGJveFxuICAgICIsIm1lcm1haWQiOiJ7XG4gIFwidGhlbWVcIjogXCJkZWZhdWx0XCJcbn0iLCJ1cGRhdGVFZGl0b3IiOmZhbHNlLCJhdXRvU3luYyI6dHJ1ZSwidXBkYXRlRGlhZ3JhbSI6dHJ1ZX0",
    "eyJjb2RlIjoiZ3JhcGggVEJcbiAgICBBW0NvbnRyaWJ1dG9yXVxuICAgIEJbSW5saW5lPGJyPjxicj5NZXJtYWlkIGNvZGU8YnI-YWRkZWQgdG8gLm1kIGZpbGVdXG4gICAgQ1tNZXJtYWlkK1NWRzxicj48YnI-QWRkIG1lcm1haWQtZ2VuZXJhdGVkPGJyPnN2ZyBmaWxlIHRvIC5tZCBmaWxlXVxuICAgIERbRXh0ZXJuYWwgdG9vbDxicj48YnI-QWRkIGV4dGVybmFsLXRvb2wtPGJyPmdlbmVyYXRlZCBzdmcgZmlsZTxicj50byAubWQgZmlsZV1cblxuICAgIEEgLS0-IEJcbiAgICBBIC0tPiBDXG4gICAgQSAtLT4gRFxuXG4gICAgY2xhc3NEZWYgYm94IGZpbGw6I2ZmZixzdHJva2U6IzAwMCxz",
    "eyJjb2RlIjoiZmxvd2NoYXJ0IExSXG4gICAgQVsxLiBVc2UgZXh0ZXJuYWw8YnI-dG9vbCB0byBjcmVhdGUvZWRpdDxicj5kaWFncmFtXVxuICAgIEJbMi4gSWYgcG9zc2libGUsIHNhdmU8YnI-ZGlhZ3JhbSBjb29yZGluYXRlczxicj5mb3IgY29udHJpYnV0b3I8YnI-YWNjZXNzXVxuICAgIENbMy4gR2VuZXJhdGUgLnN2ZyA8YnI-b3IucG5nIGZpbGU8YnI-YW5kIGRvd25sb2FkIHRvPGJyPmFwcHJvcHJpYXRlPGJyPmltYWdlcy8gZm9sZGVyXVxuICAgIHN1YmdyYXBoIHdbIF1cbiAgICBkaXJlY3Rpb24gVEJcbiAgICBEWzQuIFVzZSBmaWd1cmUgc2hvcnRjb2RlPGJyPnRvIHJlZmVyZW5j",
    "XRoIiwidGVuYW50IjoiNzJmOTg4YmYtODZmMS00MWFmLTkxYWItMmQ3Y2QwMTFkYjRhIiwidXNlcm5hbWUiOiJmb28ifQ",
    "eyJjb2RlIjoiZmxvd2NoYXJ0IExSXG4gICAgQVsxLiBVc2UgbGl2ZSBlZGl0b3I8YnI-IHRvIGNyZWF0ZS9lZGl0PGJyPmRpYWdyYW1dXG4gICAgQlsyLiBTdG9yZSBkaWFncmFtPGJyPlVSTCBzb21ld2hlcmVdXG4gICAgQ1szLiBHZW5lcmF0ZSAuc3ZnIGZpbGU8YnI-YW5kIGRvd25sb2FkIHRvPGJyPmltYWdlcy8gZm9sZGVyXVxuICAgIHN1YmdyYXBoIHdbIF1cbiAgICBkaXJlY3Rpb24gVEJcbiAgICBEWzQuIFVzZSBmaWd1cmUgc2hvcnRjb2RlPGJyPnRvIHJlZmVyZW5jZSAuc3ZnPGJyPmZpbGUgaW4gcGFnZTxicj4ubWQgZmlsZV0gLS0-XG4gICAgRVs1LiBBZGQgY2FwdGlvbl1cbiAg",
}

def main():
    if not TOKENIZER_DIR.exists():
        raise FileNotFoundError(
            f"Expanded tokenizer not found: {TOKENIZER_DIR}"
        )

    # Load the current expanded tokenizer.
    current_tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    current_vocab = current_tokenizer.get_vocab()

    # Identify added tokens, preserving their original token IDs.
    base_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base_vocab_size = len(base_tokenizer)

    added_tokens = sorted(
        (
            (token_id, token)
            for token, token_id in current_vocab.items()
            if token_id >= base_vocab_size
            and token not in TOKENS_TO_REMOVE
        ),
        key=lambda item: item[0],
    )

    removed_tokens = [
        token
        for token_id, token in sorted(
            (token_id, token)
            for token, token_id in current_vocab.items()
            if token_id >= base_vocab_size
            and token in TOKENS_TO_REMOVE
        )
    ]

    print(f"Original expanded vocabulary: {len(current_tokenizer)}")
    print(f"Added tokens before removal: {len(current_vocab) - base_vocab_size}")
    print(f"Unwanted tokens found: {len(removed_tokens)}")
    print(f"Tokens to retain: {len(added_tokens)}")

    if len(removed_tokens) != len(TOKENS_TO_REMOVE):
        missing = TOKENS_TO_REMOVE - set(removed_tokens)
        print(f"Warning: {len(missing)} specified tokens were not found.")

    # Rebuild from the base tokenizer.
    rebuilt_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokens = [token for _, token in added_tokens]
    rebuilt_tokenizer.add_tokens(tokens)

    # Verify the result.
    rebuilt_vocab = rebuilt_tokenizer.get_vocab()
    remaining_unwanted = TOKENS_TO_REMOVE.intersection(rebuilt_vocab)

    expected_size = base_vocab_size + len(added_tokens)
    if len(rebuilt_tokenizer) != expected_size:
        raise RuntimeError(
            f"Unexpected vocabulary size: {len(rebuilt_tokenizer)}; "
            f"expected {expected_size}"
        )

    if remaining_unwanted:
        raise RuntimeError(
            f"Unwanted tokens remain: {sorted(remaining_unwanted)}"
        )

    # Save to a temporary directory first, then replace the old tokenizer.
    temp_dir = TOKENIZER_DIR.with_name(
        TOKENIZER_DIR.name + "_rebuilt"
    )

    if temp_dir.exists():
        import shutil
        shutil.rmtree(temp_dir)

    rebuilt_tokenizer.save_pretrained(temp_dir)

    import shutil
    backup_dir = TOKENIZER_DIR.with_name(
        TOKENIZER_DIR.name + "_backup"
    )

    if backup_dir.exists():
        shutil.rmtree(backup_dir)

    TOKENIZER_DIR.rename(backup_dir)

    try:
        temp_dir.rename(TOKENIZER_DIR)
    except Exception:
        backup_dir.rename(TOKENIZER_DIR)
        raise

    print(f"\nRemoved {len(removed_tokens)} unwanted tokens.")
    print(f"Final vocabulary size: {len(rebuilt_tokenizer)}")
    print(f"Retained added tokens: {len(added_tokens)}")
    print(f"Updated tokenizer saved to: {TOKENIZER_DIR}")
    print(f"Backup saved to: {backup_dir}")


if __name__ == "__main__":
    main()