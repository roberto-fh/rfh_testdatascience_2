import argparse

from rfh_testdatascience_2.config import input_data, date_column
from rfh_testdatascience_2.preprocessing import Preprocessing
from rfh_testdatascience_2.model import Model


def main(
        group='>=0.8M',
        periods=24,
        test=True
):
    preprocesor = Preprocessing(
        csv_path=input_data,
        date_column=date_column
    )
    df = preprocesor.execute()

    prophet_model = Model(
        ts_df=df,
        group=group,
        periods=periods,
        test=test
    )

    prophet_model.execute()

    # model(df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train XGBoost pipeline")
    # Add parameters
    parser.add_argument(
        "--group",
        type=str,
        default="xgb",
        help="Group to forecast"
    )
    parser.add_argument(
        "--periods",
        type=int,
        default=5,
        help="Periods to forecast")
    parser.add_argument(
        "--test",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Test or prod"
    )
    args = parser.parse_args()
    main(
        group=args.group,
        periods=args.periods,
        test=args.test
    )