import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.scrolledtext import ScrolledText

from inference import analyze


class FakeNewsApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Fake News Detector")
        self.geometry("900x600")
        self.minsize(800, 500)

        # Use a nicer default font
        self.option_add("*Font", ("Segoe UI", 10))

        # Layout: left = input, right = results
        self.columnconfigure(0, weight=3)
        self.columnconfigure(1, weight=2)
        self.rowconfigure(0, weight=1)

        self._build_left_panel()
        self._build_right_panel()
        self._reset_results()

    def _build_left_panel(self):
        frame = ttk.Frame(self, padding=10)
        frame.grid(row=0, column=0, sticky="nsew")

        frame.rowconfigure(1, weight=1)
        frame.columnconfigure(0, weight=1)

        lbl = ttk.Label(frame, text="Article Text", font=("Segoe UI", 11, "bold"))
        lbl.grid(row=0, column=0, sticky="w", pady=(0, 5))

        self.text_input = ScrolledText(frame, wrap=tk.WORD, height=20)
        self.text_input.grid(row=1, column=0, sticky="nsew")

        # Buttons
        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        btn_frame.columnconfigure(2, weight=1)

        self.analyze_btn = ttk.Button(btn_frame, text="Analyze", command=self.run_analysis)
        self.analyze_btn.grid(row=0, column=0, padx=(0, 5))

        clear_btn = ttk.Button(btn_frame, text="Clear", command=self.clear_text)
        clear_btn.grid(row=0, column=1)

    def _build_right_panel(self):
        frame = ttk.Frame(self, padding=10)
        frame.grid(row=0, column=1, sticky="nsew")

        frame.rowconfigure(1, weight=0)
        frame.rowconfigure(2, weight=1)
        frame.rowconfigure(3, weight=1)
        frame.columnconfigure(0, weight=1)

        title = ttk.Label(frame, text="Analysis Result", font=("Segoe UI", 11, "bold"))
        title.grid(row=0, column=0, sticky="w", pady=(0, 5))

        # Prediction + confidence
        pred_frame = ttk.LabelFrame(frame, text="Prediction")
        pred_frame.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        pred_frame.columnconfigure(0, weight=1)

        self.pred_label = ttk.Label(pred_frame, text="No analysis yet", font=("Segoe UI", 11, "bold"))
        self.pred_label.grid(row=0, column=0, sticky="w", pady=(3, 3))

        ttk.Label(pred_frame, text="Confidence:").grid(row=1, column=0, sticky="w")

        self.conf_bar = ttk.Progressbar(pred_frame, orient="horizontal", mode="determinate")
        self.conf_bar.grid(row=2, column=0, sticky="ew", pady=(2, 0))

        self.conf_value_label = ttk.Label(pred_frame, text="0%")
        self.conf_value_label.grid(row=3, column=0, sticky="w")

        # Persuasive cues
        cues_frame = ttk.LabelFrame(frame, text="Persuasive Language Cues")
        cues_frame.grid(row=2, column=0, sticky="nsew", pady=(0, 8))
        cues_frame.rowconfigure(0, weight=1)
        cues_frame.columnconfigure(0, weight=1)

        self.cues_text = ScrolledText(cues_frame, wrap=tk.WORD, height=8)
        self.cues_text.grid(row=0, column=0, sticky="nsew")
        self.cues_text.config(state=tk.DISABLED)

        # Sentiment
        sent_frame = ttk.LabelFrame(frame, text="Sentiment")
        sent_frame.grid(row=3, column=0, sticky="nsew")
        sent_frame.columnconfigure(1, weight=1)

        ttk.Label(sent_frame, text="Positive").grid(row=0, column=0, sticky="w", pady=(3, 0))
        ttk.Label(sent_frame, text="Negative").grid(row=1, column=0, sticky="w")
        ttk.Label(sent_frame, text="Neutral").grid(row=2, column=0, sticky="w")
        ttk.Label(sent_frame, text="Compound").grid(row=3, column=0, sticky="w", pady=(5, 0))

        self.pos_bar = ttk.Progressbar(sent_frame, orient="horizontal", mode="determinate")
        self.neg_bar = ttk.Progressbar(sent_frame, orient="horizontal", mode="determinate")
        self.neu_bar = ttk.Progressbar(sent_frame, orient="horizontal", mode="determinate")

        self.pos_bar.grid(row=0, column=1, sticky="ew", padx=(5, 0), pady=(3, 0))
        self.neg_bar.grid(row=1, column=1, sticky="ew", padx=(5, 0))
        self.neu_bar.grid(row=2, column=1, sticky="ew", padx=(5, 0))

        self.compound_value_label = ttk.Label(sent_frame, text="0.00")
        self.compound_value_label.grid(row=3, column=1, sticky="w", padx=(5, 0), pady=(5, 3))

        hint = ttk.Label(
            frame,
            text="Please an article on the left and click 'Analyze'.",
            foreground="#666666",
        )
        hint.grid(row=4, column=0, sticky="w", pady=(5, 0))

    # ---- Helpers ----

    def clear_text(self):
        self.text_input.delete("1.0", tk.END)
        self._reset_results()

    def _reset_results(self):
        self.pred_label.config(text="No analysis yet", foreground="black")
        self.conf_bar["maximum"] = 100
        self.conf_bar["value"] = 0
        self.conf_value_label.config(text="0%")

        self.cues_text.config(state=tk.NORMAL)
        self.cues_text.delete("1.0", tk.END)
        self.cues_text.insert(tk.END, "No cues.")
        self.cues_text.config(state=tk.DISABLED)

        for bar in (self.pos_bar, self.neg_bar, self.neu_bar):
            bar["maximum"] = 1.0
            bar["value"] = 0

        self.compound_value_label.config(text="0.00")

    def run_analysis(self):
        text = self.text_input.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("Empty input", "Please paste or type an article first.")
            return

        try:
            self.analyze_btn.config(state=tk.DISABLED, text="Analyzing...")
            self.update_idletasks()

            # Call real model: returns (label, conf, cues, sentiment)
            label, conf, cues, sentiment = analyze(text)

            # Prediction + color
            label_str = str(label).upper()
            if label_str == "FAKE":
                color = "red"
            elif label_str == "REAL":
                color = "green"
            else:
                color = "black"

            self.pred_label.config(text=f"Prediction: {label_str}", foreground=color)

            # Confidence (0–1 → 0–100)
            try:
                conf_val = float(conf)
            except Exception:
                conf_val = 0.0
            conf_pct = max(0.0, min(1.0, conf_val)) * 100
            self.conf_bar["value"] = conf_pct
            self.conf_value_label.config(text=f"{conf_pct:.1f}%")

            # Cues (list of (span.text, span.label_))
            self.cues_text.config(state=tk.NORMAL)
            self.cues_text.delete("1.0", tk.END)
            if not cues:
                self.cues_text.insert(tk.END, "No significant persuasive cues detected.")
            else:
                lines = []
                for cue in cues:
                    # you return a tuple: (text, label)
                    try:
                        word, tag = cue
                    except Exception:
                        word, tag = str(cue), ""
                    line = f"- {word}"
                    if tag:
                        line += f"  [{tag}]"
                    lines.append(line)
                self.cues_text.insert(tk.END, "\n".join(lines))
            self.cues_text.config(state=tk.DISABLED)

            # Sentiment (VADER dict: pos, neg, neu, compound)
            if isinstance(sentiment, dict):
                pos = float(sentiment.get("pos", 0.0))
                neg = float(sentiment.get("neg", 0.0))
                neu = float(sentiment.get("neu", 0.0))
                comp = float(sentiment.get("compound", 0.0))
            else:
                pos = neg = neu = 0.0
                comp = 0.0

            self.pos_bar["value"] = max(0.0, min(1.0, pos))
            self.neg_bar["value"] = max(0.0, min(1.0, neg))
            self.neu_bar["value"] = max(0.0, min(1.0, neu))
            self.compound_value_label.config(text=f"{comp:.2f}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to analyze article:\n{e}")
        finally:
            self.analyze_btn.config(state=tk.NORMAL, text="Analyze")


if __name__ == "__main__":
    app = FakeNewsApp()
    app.mainloop()
