import pandas as pd


class Preprocessing:

    def __init__(self, csv_path, date_column, bedroom=4, threshold=800_000):
        self.csv_path = csv_path
        self.date_colum = date_column
        self.bedroom = bedroom
        self.threshold = threshold

    @staticmethod
    def load_dataset(csv_path: str, date_column: str):
        """
        Load a CSV dataset into a pandas DataFrame.

        Parameters
        ----------
        csv_path : str
            Path to the CSV file.
        date_column : str
            Name of the column to parse as datetime.

        Returns
        -------
        pd.DataFrame
            Loaded DataFrame.
        """
        df = pd.read_csv(
            csv_path,
            parse_dates=[date_column]
        )
        return df

    def create_price_group_column(self, df, bedrooms_target=4, threshold=800_000):
        """
        Create a price group column based on average postcode prices
        for properties with a given number of bedrooms.
        """
        df['year'] = df['datesold'].dt.year

        grouped = df.groupby(['year', 'postcode', 'propertyType', 'bedrooms'])['price'] \
            .agg(['mean', 'median', 'count']).reset_index()
        grouped.columns = ['year', 'postcode', 'propertyType', 'bedrooms',
                           'avg_price', 'median_price', 'num_sales']

        postcode_avg = grouped[grouped['bedrooms'] == bedrooms_target] \
            .groupby('postcode')['avg_price'].mean().reset_index()
        postcode_avg.columns = ['postcode', 'overall_avg_price']

        postcode_avg['price_group'] = postcode_avg['overall_avg_price'] \
            .apply(lambda x: f'<{threshold / 1_000_000:.1f}M' if x < threshold else f'>={threshold / 1_000_000:.1f}M')

        df = df.merge(postcode_avg[['postcode', 'price_group']], on='postcode', how='left')

        return df[df['bedrooms'] == bedrooms_target][['datesold', 'price_group', 'price']]

    @staticmethod
    def monthly_avg_price(df_price_group_time):
        """
        Calculate average monthly prices per price group.
        """
        df_price_group_time['year_month'] = df_price_group_time['datesold'].dt.to_period('M')
        monthly_avg = (
            df_price_group_time
            .groupby(['year_month', 'price_group'])['price']
            .mean()
            .reset_index()
            .rename(columns={'price': 'avg_price'})
        )
        return monthly_avg

    def execute(self):
        df_raw = self.load_dataset(self.csv_path, self.date_colum)
        df_transform = self.create_price_group_column(df_raw, self.bedroom, self.threshold)
        df_monthly = self.monthly_avg_price(df_transform)
        return df_monthly
