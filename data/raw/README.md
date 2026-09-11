# Raw Dataset

The raw dataset is intentionally not included in this repository because it is large.

## Required dataset

Download the Customer Support on Twitter dataset from Kaggle:

thoughtvector/customer-support-on-twitter

After downloading the dataset, place the main CSV file here:

data/raw/twcs.csv

The expected file structure is:

data/
└── raw/
    └── twcs.csv

The repository's data-preparation pipeline expects these columns:

- tweet_id
- author_id
- inbound
- created_at
- text
- response_tweet_id

The repository does not commit the raw dataset or generated processed datasets because of their size.