import json

INPUT_FILE = "data/OASST2/2023-11-05_oasst2_ready.trees.jsonl"
OUTPUT_FILE = "data/oasst2_en_chat.jsonl"


def extract_paths(node, current_path, all_paths):
    """
    Recursively extract all root-to-leaf paths.
    """
    # Skip deleted or non-English messages
    if node.get("deleted", False):
        return
    if node.get("lang") != "en":
        return

    role = node.get("role")
    text = node.get("text", "").strip()

    if not text:
        return

    # Convert role
    if role == "prompter":
        mapped_role = "user"
    elif role == "assistant":
        mapped_role = "assistant"
    else:
        return

    new_path = current_path + [{"role": mapped_role, "content": text}]

    replies = node.get("replies", [])

    if not replies:
        # Leaf node → save full conversation
        # Ensure it ends with assistant
        if len(new_path) >= 2 and new_path[-1]["role"] == "assistant":
            all_paths.append(new_path)
        return

    for child in replies:
        extract_paths(child, new_path, all_paths)


def main():
    total_saved = 0

    with open(INPUT_FILE, "r", encoding="utf-8") as f_in, \
         open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:

        for line in f_in:
            tree = json.loads(line)

            root = tree.get("prompt")
            if not root:
                continue

            paths = []
            extract_paths(root, [], paths)

            for p in paths:
                json.dump({"messages": p}, f_out, ensure_ascii=False)
                f_out.write("\n")
                total_saved += 1

    print(f"Saved {total_saved} English conversations to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()