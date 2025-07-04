import os

# set BCTOOLS_DEBUG to 1 to enable debug logging
os.environ.setdefault('BCTOOLS_DEBUG', '1')
from biocomptools.logging_config import get_logger, setup_logging
from sqlmodel import create_engine, Session, inspect
from typing import Dict, Annotated, Optional
import pandas as pd
from biocomp.models import Category, Part, L0, L1, L2, SequestronType, Sequestron, PartsDB
import gspread
import biocomptools.toollib.common as cm
from pathlib import Path
import time
from datetime import datetime
import sys
from gspread.exceptions import APIError, SpreadsheetNotFound
from pydantic import BaseModel, Field, model_validator
from dracon.commandline import make_program, Arg


setup_logging(force=False)
logger = get_logger(__name__)


class PartsDBUpdater(BaseModel):
    """Update biocomp parts database with Google Sheets data."""
    
    db_url: Annotated[Optional[str], Arg(help="Database URL. If not provided, uses default location.")] = None
    dry_run: Annotated[bool, Arg(help="Fetch sheets but don't update database")] = False
    google_app_credentials: Annotated[str, Arg(help="Path to Google application credentials")] = Field(
        default_factory=lambda: cm.config.db.gsheet.google_app_credentials_path
    )
    sheet_key: Annotated[str, Arg(help="Google Sheets key")] = Field(
        default_factory=lambda: cm.config.db.gsheet.parts_sheet_key
    )
    
    @model_validator(mode='after')
    def setup_db_url(self) -> 'PartsDBUpdater':
        """Set up default database URL if not provided."""
        if self.db_url is None:
            try:
                root = Path(cm.config.paths.root).expanduser().resolve()
                db_path = root / 'partsdb.sqlite'
                self.db_url = f'sqlite:///{db_path}'
                logger.info(f"Database URL not provided. Using default location:")
                logger.info(f"  Root directory: {root}")
                logger.info(f"  Database file: {db_path}")
                logger.info(f"  Database URL: {self.db_url}")
            except Exception as e:
                logger.error(f"Error determining default database path: {str(e)}")
                logger.exception(e)
                raise
        else:
            logger.info(f"Using provided database URL: {self.db_url}")
        return self

    def validate_credentials(self) -> bool:
        """Validate Google credentials file exists and is non-empty."""
        try:
            path = Path(self.google_app_credentials)
            return path.exists() and path.is_file() and path.stat().st_size > 0
        except Exception as e:
            logger.error(f"Error validating credentials file: {str(e)}")
            return False


    def get_all_google_sheets(
        self,
        first_col_as_index: bool = False,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch all sheets from a Google Sheets workbook with enhanced error handling and logging.
        """
        start_time = time.time()
        logger.info(
            f"Starting Google Sheets load operation at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )

        if not self.validate_credentials():
            raise ValueError(f"Invalid or missing credentials file: {self.google_app_credentials}")

        try:
            logger.info("Establishing connection to Google Sheets API")
            gspread_client = gspread.service_account(filename=self.google_app_credentials)
            connection_time = time.time() - start_time

            try:
                logger.info(f"Opening workbook with key: {self.sheet_key}")
                workbook = gspread_client.open_by_key(self.sheet_key)
            except SpreadsheetNotFound:
                logger.error(f"Could not find spreadsheet with key: {self.sheet_key}")
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
            logger.error(f"Fatal error in get_all_google_sheets: {str(e)}")
            logger.exception("Detailed traceback:")
            raise


    def populate_database(self, sheets_dict: Dict[str, pd.DataFrame]) -> bool:
        """Populate database with enhanced error handling and transaction support."""
        start_time = time.time()
        logger.info(f"Starting database population at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Writing database to: {self.db_url}")

        try:
            engine = create_engine(self.db_url, echo=logger.getEffectiveLevel() == 10)

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
                    # Extract the actual file path from the SQLite URL
                    if self.db_url.startswith('sqlite:///'):
                        db_file_path = self.db_url.replace('sqlite:///', '')
                        logger.info(f"Database written to: {db_file_path}")
                    return True

                except Exception as e:
                    logger.error(f"Error during database population: {str(e)}")
                    logger.info("Rolling back transaction")
                    session.rollback()
                    raise

        except Exception as e:
            logger.error(f"Fatal error in populate_database: {str(e)}")
            return False

    def run(self) -> None:
        """Execute the parts database update."""
        try:
            logger.info("Starting sheet fetch operation")
            sheets_dict = self.get_all_google_sheets()

            if self.dry_run:
                logger.info("Dry run completed - skipping database update")
                return

            if sheets_dict:
                success = self.populate_database(sheets_dict)
                if not success:
                    logger.error("Database population failed")
                    sys.exit(1)
            else:
                logger.error("Failed to fetch sheets")
                sys.exit(1)

        except Exception as e:
            logger.error(f"Fatal error in run: {str(e)}")
            sys.exit(1)


def main() -> None:
    cli_prog = make_program(
        PartsDBUpdater,
        name='biocomp-updatepartsdb',
        description='Update biocomp parts database from Google Sheets data.',
    )
    updater, _ = cli_prog.parse_args(sys.argv[1:], capture_globals=False)  # type: ignore[misc]
    updater.run()


if __name__ == "__main__":
    main()
