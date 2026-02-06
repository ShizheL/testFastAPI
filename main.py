import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

from pycontrails import Flight
from pycontrails.datalib.ecmwf import ERA5
from pycontrails.models.cocip import Cocip
from pycontrails.models.ps_model import PSFlight

load_dotenv()

CDS_API_KEY = os.getenv("ECMWF_API_KEY")
CDS_URL = "https://cds.climate.copernicus.eu/api"

cds_api_rc = f"""url: {CDS_URL}
key: {CDS_API_KEY}
"""

with open(os.path.expanduser('~/.cdsapirc'), 'w') as f:
    f.write(cds_api_rc)

app = FastAPI()

class FlightSegment(BaseModel):
    start_time: datetime
    duration_hours: float
    p1_long: float
    p1_lat: float
    p2_long: float
    p2_lat: float
    altitude_ft: float
    aircraft_type: str

@app.post("/compute-ef")
def compute_ef_segment(seg: FlightSegment):
    end_time = seg.start_time + timedelta(hours=seg.duration_hours)

    flight_data = pd.DataFrame({
        "longitude": np.array([seg.p1_long, seg.p2_long]),
        "latitude": np.array([seg.p1_lat, seg.p2_lat]),
        "altitude_ft": np.array([seg.altitude_ft, seg.altitude_ft]),
        "time": np.array([seg.start_time, seg.start_time])
    })

    flight = Flight(
        data=flight_data,
        aircraft_type=seg.aircraft_type,
        flight_id="test_flight"
    )

    era5 = ERA5(
        time=(seg.start_time, end_time),
        variables=Cocip.met_variables,
        pressure_levels = [300, 250, 225, 200],
    )
    met = era5.open_metdataset()

    era5_rad = ERA5(
        time=(seg.start_time, end_time),
        variables=Cocip.rad_variables,
    )
    rad = era5_rad.open_metdataset()

    ps_model = PSFlight()
    cocip = Cocip(
        met=met,
        rad=rad,
        aircraft_performance=ps_model
    )

    output = cocip.eval(flight)

    result_df = output.dataframe
    return {"ef": result_df['ef'].tolist()}