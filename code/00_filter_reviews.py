"""Check the image-eligibility filter before the existing analysis starts."""
import csv
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / 'input'

def read_rows(name):
    with (DATA / name).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))

def main():
    raw = read_rows('raw_reviews.csv')
    coding = read_rows('human_coding.csv')
    ids = [row['record_id'] for row in raw]
    if len(raw) != 3176 or len(set(ids)) != len(ids):
        raise ValueError('Expected 3,176 unique source reviews')
    if any(row['If pic'] not in {'Y', 'N'} for row in raw):
        raise ValueError('If pic must be Y or N')
    selected = {row['record_id'] for row in raw if row['If pic'] == 'Y'}
    coded_ids = [row['record_id'] for row in coding]
    if len(selected) != 3068 or len(coded_ids) != 3068 or set(coded_ids) != selected:
        raise ValueError('The Y subset must match the 3,068 manual-coding records exactly')
    # Existing downstream scripts consume this same eligible set and its supplied scores.
    # No scores are generated or changed by the image-eligibility filter.
    print('Image eligibility: 3,176 source reviews -> 3,068 Y; 108 N. Manual-coding IDs match.', flush=True)

if __name__ == '__main__':
    main()
