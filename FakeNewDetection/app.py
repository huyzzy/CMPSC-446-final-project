from inference import analyze

print("=== Fake News Detector ===")
print("Type or paste an article below.")
print("Press ENTER twice to analyze.\n")

while True:
    lines = []
    while True:
        line = input()
        if line.strip() == "":
            break
        lines.append(line)

    article = " ".join(lines)

    if len(article.strip()) == 0:
        print("Exiting...")
        break

    label, conf, cues = analyze(article)

    print(f"\nPrediction: {label} ({conf:.2f})")

    if cues:
        print("\nPersuasive Language Detected:")
        for word, tag in cues:
            print(f" - {word} → {tag}")
    else:
        print("\nNo significant persuasion cues detected.")

    print("\n--- Paste next article or press ENTER to quit ---\n")
