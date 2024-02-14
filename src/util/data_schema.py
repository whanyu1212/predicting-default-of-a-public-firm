import pandera as pa
from pandera import Check, Column, DataFrameSchema

# Define the schema for the data
input_data_schema = DataFrameSchema(
    {
        "CompNo": Column(str, nullable=False, coerce=True),
        "StkIndx": Column(float, nullable=True, coerce=True),
        "STInt": Column(float, nullable=False, coerce=True),
        "dtdlevel": Column(float, nullable=True, coerce=True),
        "dtdtrend": Column(float, nullable=True, coerce=True),
        "liqnonfinlevel": Column(float, nullable=True, coerce=True),
        "liqnonfintrend": Column(float, nullable=True, coerce=True),
        "ni2talevel": Column(float, nullable=True, coerce=True),
        "ni2tatrend": Column(float, nullable=True, coerce=True),
        "sizelevel": Column(float, nullable=True, coerce=True),
        "sizetrend": Column(float, nullable=True, coerce=True),
        "m2b": Column(float, nullable=True, coerce=True),
        "sigma": Column(float, nullable=True, coerce=True),
        "DTDmedianNonFin": Column(float, nullable=True, coerce=True),
        "Company_name": Column(str, nullable=False, coerce=True),
        "INDUSTRY2": Column(str, nullable=True, coerce=True),
        "Date": Column(pa.DateTime, nullable=False, coerce=True),
        "Y": Column(pa.Category, checks=Check.isin([0, 1]), nullable=False, coerce=True),
    }
)
