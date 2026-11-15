# Follow-up task: your capstone model on a second dataset

Within two weeks of the capstone day, run your capstone model again on a **second dataset** and send the link
to your repository to the trainer. The task is tracked: it completes the course.

## The second dataset

| Your capstone | The second dataset | How to get it |
|---|---|---|
| Naqsh | 8,000 photographs of the same eight patterns, taken at another workshop with another camera and light | `uv run python capstone/starter.py naqsh --second` |
| Khayt | 30 new queries, 10 English, 10 Arabic and 10 mixed, written and labelled by a colleague who has not seen your results | The colleague writes them from the catalogue; you run them once |
| Wasf | Products 21 to 40 of the catalogue, and the same task with the product line in Arabic | `uv run python capstone/starter.py wasf --second` |

## What to do

1. Before you run anything, write down what you expect: which numbers will hold, which will fall, and why.
2. Run the model of your capstone, unchanged, on the second dataset. Report the same table as in your capstone
   README, next to the first one.
3. Make **one** change that you expect to help (for Naqsh, for example, fine-tuning on 100 labelled photographs
   of the second set). Report the table again.
4. Add a section "Second dataset" to your README: the two tables, what you expected against what came out,
   three failures with a likely cause each, and what the first test did not tell you.

## What the trainer checks

- The numbers come back from a clean clone with the commands the README gives.
- The expectation was written before the run (the commit history shows it).
- The unchanged model was measured first, and one change only was made after it.
- The limits section of the README was updated with what the second dataset showed.

## AI policy

AI-full. The section is yours: you may be asked to explain any number in it.
