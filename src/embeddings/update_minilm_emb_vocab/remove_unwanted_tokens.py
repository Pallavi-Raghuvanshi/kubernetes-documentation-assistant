# src/embeddings/update_minilm_emb_vocab/remove_unwanted_tokens.py
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]

BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOKENIZER_DIR = ROOT / "models" / "kubernetes_updated_tokenizer_vocab"

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