from biocomptools.logging_config import get_logger, setup_logging
from sqlmodel import create_engine, Session, inspect
from typing import Dict
import pandas as pd
from biocomp.models import Category, Part, L0, L1, L2, SequestronType, Sequestron, PartsDB
import gspread
import biocomptools.toollib.common as cm
import argparse
from pathlib import Path
import time
from datetime import datetime
import sys
from gspread.exceptions import APIError, SpreadsheetNotFound


GOOGLE_APP_CREDENTIALS = cm.config.db.gsheet.google_app_credentials_path
SHEET_KEY = cm.config.db.gsheet.parts_sheet_key

logger = get_logger(__name__)


def validate_credentials(credentials_path: str) -> bool:
    try:
        path = Path(credentials_path)
        return path.exists() and path.is_file() and path.stat().st_size > 0
    except Exception as e:
        logger.error(f"Error validating credentials file: {str(e)}")
        return False


def getAllGoogleSheets(
    key: str = SHEET_KEY,
    credentials: str = GOOGLE_APP_CREDENTIALS,
    first_col_as_index: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    Fetch all sheets from a Google Sheets workbook with enhanced error handling and logging.
    """
    start_time = time.time()
    logger.info(
        f"Starting Google Sheets load operation at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    if not validate_credentials(credentials):
        raise ValueError(f"Invalid or missing credentials file: {credentials}")

    try:
        logger.info("Establishing connection to Google Sheets API")
        gspread_client = gspread.service_account(filename=credentials)
        connection_time = time.time() - start_time

        try:
            logger.info(f"Opening workbook with key: {key}")
            workbook = gspread_client.open_by_key(key)
        except SpreadsheetNotFound:
            logger.error(f"Could not find spreadsheet with key: {key}")
            raise
        except APIError as e:
            logger.error(f"API error accessing spreadsheet: {str(e)}")
            raise

        sheets = workbook.worksheets()
        logger.info(
            f"Found {len(sheets)} sheets in workbook: {', '.join(sheet.title for sheet in sheets)}"
        )

        sheets_dict = {}
        stats = {
            'total_rows_processed': 0,
            'total_empty_rows': 0,
            'sheets_with_errors': [],
            'empty_sheets': [],
        }

        for sheet in sheets:
            sheet_start_time = time.time()
            logger.info(f"Processing sheet: {sheet.title}")

            try:
                raw_data = sheet.get_all_records()
                logger.debug(f"Retrieved {len(raw_data)} raw records from {sheet.title}")

                df = pd.DataFrame(raw_data)
                if df.empty:
                    logger.warning(f"Sheet {sheet.title} is empty")
                    stats['empty_sheets'].append(sheet.title)
                    continue

                logger.debug(f"Column names in {sheet.title}: {', '.join(df.columns)}")

                null_counts = df.isnull().sum()
                if null_counts.any():
                    logger.warning(f"Found null values in {sheet.title}:")
                    for col, count in null_counts[null_counts > 0].items():
                        logger.warning(f"  - {col}: {count} null values")

                col0 = df.columns[0]

                previous_len = len(df)
                df = df[df[col0] != ""]
                empty_rows = previous_len - len(df)
                stats['total_empty_rows'] += empty_rows
                stats['total_rows_processed'] += len(df)

                logger.debug(
                    f"Sheet {sheet.title}: Removed {empty_rows} empty rows, kept {len(df)} rows"
                )

                if first_col_as_index:
                    df.set_index(col0, inplace=True)
                    logger.debug(f"Set {col0} as index for {sheet.title}")

                sheets_dict[sheet.title] = df
                sheet_time = time.time() - sheet_start_time
                logger.info(f"Completed processing {sheet.title} in {sheet_time:.2f} seconds")

            except Exception as e:
                logger.error(f"Error processing sheet {sheet.title}: {str(e)}")
                stats['sheets_with_errors'].append(sheet.title)
                continue

        total_time = time.time() - start_time
        logger.info("---Operation Summary ---")
        logger.info(f"Total time: {total_time:.2f} seconds")
        logger.info(f"Processed {stats['total_rows_processed']} total rows")
        logger.info(f"Removed {stats['total_empty_rows']} total empty rows")
        logger.info(f"Successfully processed sheets: {len(sheets_dict)}/{len(sheets)}")

        if stats['empty_sheets']:
            logger.warning(f"Empty sheets: {', '.join(stats['empty_sheets'])}")
        if stats['sheets_with_errors']:
            logger.error(f"Sheets with errors: {', '.join(stats['sheets_with_errors'])}")

        return sheets_dict

    except Exception as e:
        logger.error(f"Fatal error in getAllGoogleSheets: {str(e)}")
        logger.exception("Detailed traceback:")
        raise


def populate_database(sheets_dict: Dict[str, pd.DataFrame], db_url: str) -> bool:
    """Populate database with enhanced error handling and transaction support."""
    start_time = time.time()
    logger.info(f"Starting database population at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        engine = create_engine(db_url, echo=logger.getEffectiveLevel() == 10)

        logger.info("Creating database schema...")
        PartsDB.metadata.drop_all(engine)  # clear existing tables
        PartsDB.metadata.create_all(engine)
        inspector = inspect(engine)
        table_names = inspector.get_table_names()
        logger.info(f"Created tables: {table_names}")

        models_to_update = [
            (Category, 'categories'),
            (Part, 'parts'),
            (L0, 'L0s'),
            (L1, 'L1s'),
            (L2, 'L2s'),
            (SequestronType, 'sequestron_types'),
            (Sequestron, 'sequestrons'),
        ]

        with Session(engine) as session:
            try:
                logger.info("Starting database transaction")

                for model, sheet_name in models_to_update:
                    if sheet_name not in sheets_dict:
                        logger.error(f"Missing required sheet: {sheet_name}")
                        raise ValueError(f"Missing required sheet: {sheet_name}")

                    df = sheets_dict[sheet_name]
                    logger.info(f"Populating {model.__name__} with {len(df)} records")

                    for _, row in df.iterrows():
                        try:
                            session.add(model(**row.to_dict()))
                        except Exception as e:
                            logger.error(f"Error adding {model.__name__} record: {str(e)}")
                            raise

                session.commit()

                total_time = time.time() - start_time
                logger.info(
                    f"Database population completed successfully in {total_time:.2f} seconds"
                )
                return True

            except Exception as e:
                logger.error(f"Error during database population: {str(e)}")
                logger.info("Rolling back transaction")
                session.rollback()
                raise

    except Exception as e:
        logger.error(f"Fatal error in populate_database: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Update biocomp database with google sheets data")
    parser.add_argument("db_url", type=str, help="Database URL", nargs='?', default=None)
    parser.add_argument(
        "--log-level",
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help="Set the logging level",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Fetch sheets but don't update database"
    )

    args = parser.parse_args()
    logger.setLevel(args.log_level)

    if args.db_url is None:
        try:
            root = Path(cm.config.paths.root).expanduser().resolve()
            args.db_url = f'sqlite:///{root}/partsdb.sqlite'
            logger.info(f"Database URL/path not provided. Using default: {args.db_url}")
        except Exception as e:
            logger.error(f"Error determining default database path: {str(e)}")
            sys.exit(1)

    try:
        logger.info("Starting sheet fetch operation")
        sheets_dict = getAllGoogleSheets()

        if args.dry_run:
            logger.info("Dry run completed - skipping database update")
            return

        if sheets_dict:
            success = populate_database(sheets_dict, args.db_url)
            if not success:
                logger.error("Database population failed")
                sys.exit(1)
        else:
            logger.error("Failed to fetch sheets")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Fatal error in main: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
