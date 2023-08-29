import pyTMD.io
import pyTMD.time
import pyTMD.utilities
import pyTMD.predict

import datetime

import numpy as np

def get_pytmd_tide_timeseries(list_datetimes, latitude, longitude, path_to_arc2kmtm="/home/jrmet/Desktop/Data/tides/resource_map_doi_10_18739_A2D21RK6K/"):
    model_path = path_to_arc2kmtm
    model_name = "Arc2kmTM"
    model_is_compressed = False
    model_atlas = "netcdf"
    lat_to_predict = latitude
    lon_to_predict = longitude
    
    pytmd_times_to_predict = [pyTMD.time.convert_calendar_dates(crrt_time.year, crrt_time.month, crrt_time.day, crrt_time.hour, crrt_time.minute, crrt_time.second) for crrt_time in list_datetimes]
    model_time_delta = np.zeros_like(pytmd_times_to_predict)

    dict_results = {}
    dict_results["list_datetimes"] = list_datetimes

    ##############################################################
    # tide elevation stuff
    tide_model_base = pyTMD.io.model(
        model_path,
        format=model_atlas,
        compressed=model_is_compressed,
    )
    
    tide_model = tide_model_base.elevation(model_name)
    # tide_model.type is type 'z', i.e. just tide elevation and a single output
    
    tide_constituents = pyTMD.io.OTIS.read_constants(
        tide_model.grid_file,
        tide_model.model_file,
        tide_model.projection,
        type=tide_model.type,
        grid=tide_model.format
    )
    
    list_time_constituents = tide_constituents.fields
    print(f"tidal constituents used in the present model are: {list_time_constituents = }")
    
    tide_amplitude, tide_phase, model_bathymetry = pyTMD.io.OTIS.interpolate_constants(
        np.atleast_1d(lon_to_predict),
        np.atleast_1d(lat_to_predict),
        tide_constituents,
        tide_model.projection,
        type=tide_model.type,
        method='spline',
        extrapolate=True
    )
    
    tide_complex_phase = -1j * tide_phase * np.pi / 180.0
    tide_constituent_oscillation = tide_amplitude * np.exp(tide_complex_phase)
    
    tide_meters = pyTMD.predict.time_series(
        np.atleast_1d(pytmd_times_to_predict),
        tide_constituent_oscillation,
        list_time_constituents,
        deltat=model_time_delta,
        corrections=tide_model.format
    )
    
    minor_constituents = pyTMD.predict.infer_minor(
        np.atleast_1d(pytmd_times_to_predict),
        tide_constituent_oscillation,
        list_time_constituents,
        deltat=model_time_delta,
        corrections=tide_model.format)
    
    tide_meters.data[:] += minor_constituents.data[:]

    dict_results["tide_meters"] = tide_meters

    ##############################################################
    # current stuff
    
    current_model = tide_model_base.current(model_name)
    # current_model.type is type ['u', 'v'], i.e. BOTH u and v components of the current need to be retrieved
    
    tide_amplitude = {}
    tide_phase = {}
    model_bathymetry = {}
    tide_constituents = {}
    tide_complex_phase = {}
    tide_constituent_oscillation = {}
    
    for type in current_model.type:
        tide_amplitude[type], tide_phase[type], model_bathymetry[type], tide_constituents[type] = pyTMD.io.OTIS.extract_constants(
            np.atleast_1d(lon_to_predict),
            np.atleast_1d(lat_to_predict),
            current_model.grid_file,
            current_model.model_file['u'],
            current_model.projection,
            type=type,
            method='spline',
            grid=current_model.format
        )
        
        tide_complex_phase[type] = -1j * tide_phase[type] * np.pi / 180.0
        tide_constituent_oscillation[type] = tide_amplitude[type] * np.exp(tide_complex_phase[type])
    
    tide_current_major = {}
    tide_current_minor = {}
    tide_current = {}
    
    for type in current_model.type:
        tide_current_major[type] = pyTMD.predict.time_series(
            np.atleast_1d(pytmd_times_to_predict),
            tide_constituent_oscillation[type],
            tide_constituents[type],
            #detlat=0,  # this is true for the present model, but maybe not for all of them!
            corrections=current_model.format
        )
        
        tide_current_minor[type] = pyTMD.predict.infer_minor(
            np.atleast_1d(pytmd_times_to_predict),
            tide_constituent_oscillation[type],
            tide_constituents[type],
            #detlat=0,  # this is true for the present model, but maybe not for all of them!
            corrections=current_model.format
        )
        
        tide_current[type] = tide_current_major[type] + tide_current_minor[type]
    
    dict_results["tide_current_u_cm_per_s"] = tide_current["u"]
    dict_results["tide_current_v_cm_per_s"] = tide_current["v"]
    
    return dict_results
