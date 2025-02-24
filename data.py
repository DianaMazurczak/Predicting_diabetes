import pandas as pd
from ucimlrepo import fetch_ucirepo


def main(id):
    # fetch dataset
    cdc_diabetes_health_indicators = fetch_ucirepo(id=id)

    # data (as pandas dataframes)
    X = cdc_diabetes_health_indicators.data.features
    y = cdc_diabetes_health_indicators.data.targets

    # variable information
    information = pd.DataFrame(cdc_diabetes_health_indicators.variables)
    print(information.columns)
    information['demographic']


if __name__ == '__main__':
    main(id=891)
