"""Quick sanity check on style_features.csv."""
import csv

rows = list(csv.DictReader(open("reports/style_features.csv")))

early = [r for r in rows if r["pub_year"] and 1967 <= int(r["pub_year"]) <= 1984]
late  = [r for r in rows if r["pub_year"] and 2016 <= int(r["pub_year"]) <= 2026]

def avg(lst, col):
    vals = [float(r[col]) for r in lst if r[col]]
    return sum(vals)/len(vals) if vals else 0

metrics = [
    "sent_mean_words", "sent_pct_long", "sent_pct_short",
    "ttr", "abstract_ratio", "mean_word_len",
    "hedge_per_1k", "assert_per_1k",
    "i_per_1k", "we_per_1k", "passive_per_1k",
    "causal_per_1k", "contrast_per_1k",
    "citation_per_1k", "self_cite_per_1k",
]

print("=== EARLY (1967-1984) vs LATE (2016-2026) ===")
print(f"\n{'Metric':<26} {'Early(n=' + str(len(early)) + ')':>12} {'Late(n=' + str(len(late)) + ')':>12}  {'Δ':>7}")
print("-"*60)
for m in metrics:
    e = avg(early, m); l = avg(late, m)
    print(f"{m:<26} {e:>12.3f} {l:>12.3f}  {l-e:>+7.3f}")

print("\n=== SAMPLE ROWS (first 8 by year) ===")
hdr = f"{'Year':<5} {'Bk':>3} {'Ch':>3} {'Words':>6} {'SMW':>5} {'TTR':>5} {'Hdg':>5} {'Ass':>5} {'I/k':>5} {'We/k':>5}  Title"
print(hdr)
print("-"*105)
for r in rows[:8]:
    print(f"{r['pub_year']:<5} {r['book_number']:>3} {r['chapter_number']:>3} "
          f"{int(r['narrative_words']):>6} "
          f"{float(r['sent_mean_words']):>5.1f} "
          f"{float(r['ttr']):>5.3f} "
          f"{float(r['hedge_per_1k']):>5.2f} "
          f"{float(r['assert_per_1k']):>5.2f} "
          f"{float(r['i_per_1k']):>5.2f} "
          f"{float(r['we_per_1k']):>5.2f}  "
          f"{r['chapter_title'][:45]}")
