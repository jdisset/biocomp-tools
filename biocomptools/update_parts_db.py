from sqlmodel import SQLModel, create_engine, Session, select
from typing import Dict, Any, List
import pandas as pd
import biocomp as bc
from biocomp.models import Category, Part, L0, L1, L2, SequestronType, Sequestron
import gspread
import pandas as pd
import biocomptools.toollib.common as cm
import argparse
from pathlib import Path
import biocomp.models as bm
from biocomp.models import buildLibFromDatabase


### {{{                   --     google sheet helpers     --


GOOGLE_APP_CREDENTIALS = cm.config.db.gsheet.google_app_credentials_path
SHEET_KEY = cm.config.db.gsheet.parts_sheet_key

log = cm.get_logger("parts_sheet_loader")

def find_duplicates(df, col):
    return df[df.duplicated(subset=col, keep=False)]

def getAllGoogleSheets(key=SHEET_KEY, credentials=GOOGLE_APP_CREDENTIALS, first_col_as_index=False):
    log.info("Loading google sheets")
    gspread_client = gspread.service_account(filename=credentials)
    workbook = gspread_client.open_by_key(key)
    sheets = workbook.worksheets()
    sheets_dict = {}
    for sheet in sheets:
        log.debug(f"Loading sheet: {sheet.title}")
        df = pd.DataFrame(sheet.get_all_records())
        col0 = df.columns[0]
        previous_len = len(df)
        df = df[df[col0] != ""] # remove empty rows
        log.debug(f"Ignoring {previous_len - len(df)} empty rows from {sheet.title}")
        if first_col_as_index:
            df.set_index(col0, inplace=True)
        sheets_dict[sheet.title] = df
    return sheets_dict


def listGoogleSpreadsheets(credentials=GOOGLE_APP_CREDENTIALS):
    gspread_client = gspread.service_account(filename=credentials)
    spreadsheets = gspread_client.openall()
    if spreadsheets:
        print("Available spreadsheet workbooks:")
        for spreadsheet in spreadsheets:
            print("Title:", spreadsheet.title, "URL:", spreadsheet.url)
    else:
        print("No spreadsheets available")
        print("Please share the spreadsheet with Service Account email")



def getLibFromGoogleSheet(key=SHEET_KEY, credentials=GOOGLE_APP_CREDENTIALS):
    l = getAllGoogleSheets(key, credentials, first_col_as_index=True)

    lib = bc.PartsLibrary(
        l['parts'], l['L0s'], l['L1s'], l['L2s'], l['categories'], l['sequestrons'], l['sequestron_types']
    )

    return lib


###                                                                            }}}

## {{{                        --     sheet to db functions    --



def populate_database(sheets_dict: Dict[str, pd.DataFrame], db_url: str):
    engine = create_engine(db_url)
    SQLModel.metadata.create_all(engine)

    with Session(engine) as session:
        # List of models to update
        models_to_update = [Category, Part, L0, L1, L2, SequestronType, Sequestron]

        # Delete existing entries
        for model in models_to_update:
            session.exec(model.__table__.delete())

        # Populate Categories
        for _, row in sheets_dict['categories'].iterrows():
            session.add(Category(**row.to_dict()))

        # Populate Parts
        for _, row in sheets_dict['parts'].iterrows():
            session.add(Part(**row.to_dict()))

        # Populate L0s
        for _, row in sheets_dict['L0s'].iterrows():
            session.add(L0(**row.to_dict()))

        # Populate L1s
        for _, row in sheets_dict['L1s'].iterrows():
            session.add(L1(**row.to_dict()))

        # Populate L2s
        for _, row in sheets_dict['L2s'].iterrows():
            session.add(L2(**row.to_dict()))

        # Populate SequestronTypes
        for _, row in sheets_dict['sequestron_types'].iterrows():
            session.add(SequestronType(**row.to_dict()))

        # Populate Sequestrons
        for _, row in sheets_dict['sequestrons'].iterrows():
            session.add(Sequestron(**row.to_dict()))

        session.commit()



##────────────────────────────────────────────────────────────────────────────}}}

root = Path(cm.config.paths.root).expanduser().resolve()
db_url = f'sqlite:///{root}/partsdb.sqlite'
lib = buildLibFromDatabase(db_url)
lib.L1s

glib = getLibFromGoogleSheet()

glib.parts
lib.parts

lib.L0s
glib.L0s

glib.L1s


bc.utils.DEFAULT_LIB_PATH

sheets_dict = getAllGoogleSheets()
d = sheets_dict['L1s'].iloc[0].to_dict()

bm.L1(**d)


populate_database(sheets_dict, db_url)

def main():
    parser = argparse.ArgumentParser(description="Update biocomp database with google sheets data")
    parser.add_argument("db_url", type=str, help="Database URL", default=None)

    args = parser.parse_args()
    if args.db_url is None:
        root = Path(cm.config.paths.root).expanduser().resolve()
        args.db_url = f'sqlite:///{root}/partsdb.sqlite'
        print(f"Database URL/path not provided. Using default: {args.db_url}")

    sheets_dict = getAllGoogleSheets()
    populate_database(sheets_dict, args.db_url)

