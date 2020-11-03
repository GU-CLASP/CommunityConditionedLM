"""
Queries the pushshift.io data on BigQuery to create author/community cooccurance counts for 2015.
The results are saved to a Google Storange bucket and can be downloaded as follows:
    gsutil -m cp gs://bill-gu-research/CondLM/2015_author_sub_counts-*.csv ./data/

See: https://cloud.google.com/docs/authentication/getting-started#auth-cloud-implicit-python
"""

import click
import os
import google.auth
from google.cloud import bigquery
from google.cloud import bigquery_storage_v1beta1
from google.cloud.exceptions import NotFound


# Processes ~15GB of data (1TB free as of 11/2020 )
query = """
SELECT author, subreddit, count(*) as comment_count
FROM `fh-bigquery.reddit_comments.2015_*`
GROUP BY author, subreddit
"""

@click.command()
@click.argument('google_credentials_file', type=click.Path(exists=True))
@click.option('--google-project', type=str, default="bill-gu-research")
@click.option('--google-dataset-id', type=str, default="reddit_sample")
def cli(google_credentials_file, google_project, google_dataset_id):
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = google_credentials_file
    credentials, project_id = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    bq = bigquery.Client(credentials=credentials, project=project_id)
    bq_storage = bigquery_storage_v1beta1.BigQueryStorageClient(credentials=credentials)

    table_id = f"2015_author_sub_counts"
    dataset_ref = bq.dataset(google_dataset_id, project=google_project)
    table_ref = dataset_ref.table(table_id)

    try:
        bq.get_table(table_ref)  # Make an API request.
        print(f"{table_ref} already exists")
    except NotFound:
        print(f"Creating f{table_ref}")
        table = bigquery.Table(table_ref)
        bq.create_table(table)

    job_config = bigquery.QueryJobConfig(
        allow_large_results=True,
        destination=table_ref,
        use_legacy_sql=False,
        write_disposition='WRITE_TRUNCATE'
    )

    query_job = bq.query(query, job_config=job_config)  # Make an API request.

    destination_uri = "gs://{}/{}/{}".format(
        google_project, 'CondLM', f"{table_id}-*.csv"
    )
    extract_job = bq.extract_table(
        table_ref,
        destination_uri,
        location="US",  # Location must match that of the source table.
    )  # API request
    extract_job.result()  # Waits for job to complete.

    print(f"Exported {google_project}:{google_dataset_id}.{table_id} to {destination_uri}")

if __name__ == '__main__':
    cli()
