from src.eda import run_eda
from src.extract import fetch_models_from_api
from src.load import load_data, read_from_database
from src.schema import validate_records
from src.transform import transform_models_data


def main():
    print("=" * 65)
    print("  OPENLLM PULSE: AUTOMATED ETL + EDA PIPELINE")
    print("=" * 65)

    # Step 1: Extract raw JSON data across 4 task categories (~400 models)
    raw_records = fetch_models_from_api(limit_per_category=100)

    # Step 2: Validate schema with Pydantic & Transform with Pandas
    validated_records = validate_records(raw_records)
    cleaned_df = transform_models_data(validated_records)

    # Step 3: Load structured data into SQLite Database and CSV file
    load_data(cleaned_df)

    # Step 4: Read from SQLite Database and execute Exploratory Data Analysis
    db_df = read_from_database()
    run_eda(db_df)

    print("[SUCCESS] OpenLLM Pulse ETL + EDA Pipeline finished successfully!")


if __name__ == "__main__":
    main()