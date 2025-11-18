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

    label, conf, cues, sentiment = analyze(article)

    # Confidence Bar
from inference import confidence_bar, highlight_cues
print("\nPrediction:", label)
print("Confidence:")
print(confidence_bar(conf))

# Persuasion Cue Highlighting
if cues:
    print("\nPersuasive Language Detected:")
    for word, tag in cues:
        print(f" - {word} → {tag}")

    print("\nArticle with Highlights:")
    print(highlight_cues(article, cues))
else:
    print("\nNo significant persuasion cues detected.")

    print(f"\nPrediction: {label} ({conf:.2f})")

    if cues:
        print("\nPersuasive Language Detected:")
        for word, tag in cues:
            print(f" - {word} → {tag}")
    else:
        print("\nNo significant persuasion cues detected.")

    print(f"\nSentiment Analysis:")
    print(f" - Positive: {sentiment['pos']:.2f}")
    print(f" - Negative: {sentiment['neg']:.2f}")
    print(f" - Neutral: {sentiment['neu']:.2f}")
    print(f" - Compound: {sentiment['compound']:.2f}")

    print("\n--- Paste next article or press ENTER to quit ---\n")
