import pandas as pd
import os
import google.auth
from google.cloud import bigquery
from google.cloud import bigquery_storage_v1beta1

def remove_nul(filename):
    with open(filename, 'r') as f:
        data = f.read()
    data = data.replace('\00', '')
    with open(filename, 'w') as f:
        f.write(data)

def get_comment_counts(year, month, min_posts, bq_client, bqstorage_client):
    query_string = f"""
    SELECT subreddit, post_count
    FROM (
      SELECT subreddit, COUNT(id) as post_count
      FROM `fh-bigquery.reddit_comments.{year}_{month:02d}`
      GROUP BY subreddit
    )
    WHERE post_count > {min_posts}
    ORDER BY post_count DESC
    """
    df = bq_client.query(query_string)\
        .result()\
	.to_dataframe(bqstorage_client=bqstorage_client)
    df = df.set_index('subreddit')
    df = df.rename({'post_count': f"{year}-{month:02d}"}, axis=1)
    return df

def sample_subreddit_comments(year, month, subs, n_comments,
	gcp_project, gs_bucket, gs_dir, bq_dataset_id, bq_client):
    # bq stratified sample https://stackoverflow.com/a/52901452
    query_string = f"""
    WITH table AS (
      SELECT *, subreddit category
      FROM `fh-bigquery.reddit_comments.{year}_{month:02d}` a
    ), table_stats AS (
      SELECT *, SUM(c) OVER() total
      FROM (
	SELECT category, COUNT(*) c
	FROM table
	GROUP BY 1
	HAVING category in ({', '.join([f"'{sub}'" for sub in subs])})
        )
    )
    SELECT r.id, r.link_id, r.parent_id, r.created_utc, r.subreddit, r.author, r.body
    FROM (
      SELECT ARRAY_AGG(a ORDER BY RAND() LIMIT {n_comments}) comment
      FROM table a
      JOIN table_stats b
      USING(category)
      GROUP BY category
    ), UNNEST(comment) as r
    """
    # Query comments for the selecetd subs and save to a BQ table
    table_id = f"{gcp_project}.{bq_dataset_id}.{year}_{month:02d}"
    job_config = bigquery.QueryJobConfig(
	allow_large_results=True,
	destination=table_id,
	use_legacy_sql=False,
	write_disposition='WRITE_TRUNCATE'
    )
    # Start the query, passing in the extra configuration.
    query_job = bq_client.query(query_string, job_config=job_config)  # Make an API request.
    query_job.result()  # Wait for the job to complete.
    print(f"Comment query results saved to {table_id}.")
    # Save the data to Google Storage
    table_id = f"{year}_{month:02d}"
    destination_uri = f"gs://{gs_bucket}/{gs_dir}/{year}-{month:02d}-*.csv"
    dataset_ref = bq_client.dataset(bq_dataset_id, gcp_project)
    table_ref = dataset_ref.table(table_id)
    extract_job = bq_client.extract_table(
	table_ref,
	destination_uri,
	location="US",  # Location must match that of the source table.
    )  # API request
    extract_job.result()  # Waits for job to complete.
    print(f"Exported {gcp_project}:{bq_dataset_id}.{table_id} to {destination_uri}")


if __name__ == '__main__':

    """
    We want to find subreddits that have at least 5000 comments per month 
    for every month of 2015.
    We will sample of 250 subreddits from this list.
    And sample 5000 comments per month = 60000 comments per subreddit
    Which is 60000 * 250 = 15,000,000 comments total
    """

    google_credentials_file = '/home/xnobwi/.google-cloud/bill-gu-research-177baaa5bef7.json'
    gcp_project = 'bill-gu-research'
    gs_bucket =  'bill-gu-research'
    gs_dir = 'CondLM'
    bq_dataset_id = 'condlm_reddit_sample'

    # Make Google API clients.
    # You have to run `gcloud init` before this will work.
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = google_credentials_file
    credentials, project_id = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    bq = bigquery.Client(credentials=credentials, project=project_id)
    bq_storage = bigquery_storage_v1beta1.BigQueryStorageClient(credentials=credentials)

    df = pd.DataFrame()
    for month in range(1,13):
        df_month = get_comment_counts(2015, month, 5000, bq, bq_storage)
        df = df.merge(df_month, how='outer', left_index=True, right_index=True)

    df.to_csv('2015_subreddit_comment_counts.csv')
    # df = pd.read_csv('2015_subreddit_comment_counts.csv').set_index('subreddit')

    min_posts_per_month = 10000
    df = df.where(df >= min_posts_per_month).dropna()

    subs = list(df.index)
    n_sample_comments = 5000
    for month in range(1,13):
        sample_subreddit_comments(2015, month, subs, n_sample_comments, 
            gcp_project, gs_bucket, gs_dir, bq_dataset_id, bq)

