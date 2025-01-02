import numpy as np
from sgp4.api import Satrec, SGP4_ERRORS
import requests
import math
import json

def get_all_station_tles():
    url = "https://celestrak.org/NORAD/elements/stations.txt"
    response = requests.get(url)
    data = response.text.splitlines()

    station_data = []
    i = 0
    while i < len(data):
        name = data[i].strip()
        if i + 2 < len(data):
            line1 = data[i + 1].strip()
            line2 = data[i + 2].strip()
            if line1.startswith("1 ") and line2.startswith("2 "):
                station_data.append({
                    "name": name,
                    "line1": line1,
                    "line2": line2
                })
        i += 3
    return station_data

def get_orbital_elements_from_tle(tle_line1, tle_line2):
    satellite = Satrec.twoline2rv(tle_line1, tle_line2)
    t = satellite.jdsatepoch
    error_code, r, v = satellite.sgp4(t, 0.0)

    if error_code != 0:
        return {}

    r = np.array(r, dtype=float)
    v = np.array(v, dtype=float)
    mu = 398600.4418
    r_norm = np.linalg.norm(r)
    v_norm = np.linalg.norm(v)
    h_vec = np.cross(r, v)
    h = np.linalg.norm(h_vec)
    e_vec = (1/mu) * ((v_norm**2 - mu/r_norm)*r - np.dot(r, v)*v)
    e = np.linalg.norm(e_vec)
    energy = 0.5*(v_norm**2) - mu/r_norm
    if abs(energy) < 1e-10:
        a = r_norm
    else:
        a = -mu/(2*energy)
    hz = h_vec[2]
    inclination = math.degrees(math.acos(hz/h))
    K = np.array([0, 0, 1])
    n_vec = np.cross(K, h_vec)
    n = np.linalg.norm(n_vec)
    if n != 0:
        raan = math.degrees(math.acos(n_vec[0] / n))
        if n_vec[1] < 0:
            raan = 360 - raan
    else:
        raan = 0.0
    if e > 1e-8 and n != 0:
        omega = math.degrees(math.acos(np.dot(n_vec, e_vec)/(n*e)))
        if e_vec[2] < 0:
            omega = 360 - omega
    else:
        omega = 0.0
    if e > 1e-8:
        nu = math.degrees(math.acos(np.dot(e_vec, r)/(e*r_norm)))
        if np.dot(r, v) < 0:
            nu = 360 - nu
    else:
        if n*r_norm != 0:
            cp = np.cross(n_vec, r)
            if np.dot(cp, h_vec) >= 0:
                nu = math.degrees(math.acos(np.dot(n_vec, r)/(n*r_norm)))
            else:
                nu = 360 - math.degrees(math.acos(np.dot(n_vec, r)/(n*r_norm)))
        else:
            nu = 0.0
    return {
        "semi_major_axis_km": a,
        "eccentricity": e,
        "inclination_deg": inclination,
        "raan_deg": raan,
        "arg_of_perigee_deg": omega,
        "true_anomaly_deg": nu,
        "position_km": r.tolist(),
        "velocity_km_s": v.tolist()
    }

def construct_cw_matrix(n):
    A = np.array([
        [0,   0,   0,   1,   0,   0],
        [0,   0,   0,   0,   1,   0],
        [0,   0,   0,   0,   0,   1],
        [3*n**2, 0,   0,   0,   2*n, 0],
        [0,   0,   0,  -2*n, 0,   0],
        [0,   0,  -n**2, 0,   0,   0]
    ])
    return A

def diagonalize_matrix(A):
    w, V = np.linalg.eig(A)
    return w, V

def optimize_transfer(A, X0, Xf, tf):
    try:
        w, V = diagonalize_matrix(A)
        V_inv = np.linalg.pinv(V)
        X0_d = V_inv @ X0
        Xf_d = V_inv @ Xf
        exp_diag = np.diag(np.exp(w * tf))
        Xf_free_d = exp_diag @ X0_d
        dX_d = Xf_d - Xf_free_d
        dX = np.real(V @ dX_d)
        dv_vec_km_s = dX[3:6]
        delta_v_m_s = float(np.linalg.norm(dv_vec_km_s) * 1000.0)
        return delta_v_m_s, dX.tolist()
    except np.linalg.LinAlgError:
        print("Warning: Linear algebra error encountered. Using fallback values.")
        return 0.0, [0.0]*6
    except Exception as e:
        print(f"Warning: General error in optimization: {str(e)}")
        return 0.0, [0.0]*6

def main():
    print("===================================================")
    print("   Optimizing Small Orbital Adjustments for ALL   ")
    print("             Stations in 'stations.txt'           ")
    print("===================================================")
    station_tles = get_all_station_tles()
    if not station_tles:
        print("Could not retrieve station TLEs. Exiting.")
        return
    results = []
    X0 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    Xf = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    tf = 900.0
    mu = 398600.4418
    for station in station_tles:
        name = station["name"]
        line1 = station["line1"]
        line2 = station["line2"]
        try:
            elements = get_orbital_elements_from_tle(line1, line2)
            if not elements:
                results.append({
                    "name": name,
                    "line1": line1,
                    "line2": line2,
                    "error": "SGP4 propagation error"
                })
                continue
            a = elements["semi_major_axis_km"]
            n = math.sqrt(mu / (a**3))
            A = construct_cw_matrix(n)
            delta_v_m_s, dX = optimize_transfer(A, X0, Xf, tf)
            station_result = {
                "name": name,
                "line1": line1,
                "line2": line2,
                "orbital_elements": {
                    "semi_major_axis_km": elements["semi_major_axis_km"],
                    "eccentricity": elements["eccentricity"],
                    "inclination_deg": elements["inclination_deg"],
                    "raan_deg": elements["raan_deg"],
                    "arg_of_perigee_deg": elements["arg_of_perigee_deg"],
                    "true_anomaly_deg": elements["true_anomaly_deg"]
                },
                "mean_motion_rad_s": float(n),
                "transfer_optimization": {
                    "initial_state_km_km_s": X0.tolist(),
                    "final_state_km_km_s": Xf.tolist(),
                    "time_of_flight_s": tf,
                    "delta_v_m_s": delta_v_m_s,
                    "state_correction_km_km_s": dX
                }
            }
            results.append(station_result)
        except Exception as e:
            print(f"Error processing station {name}: {str(e)}")
            results.append({
                "name": name,
                "error": f"Processing error: {str(e)}"
            })
    with open("optimized_stations.json", "w") as f:
        json.dump(results, f, indent=4)
    print(f"Processed {len(station_tles)} station(s).")
    print("Results saved to 'optimized_stations.json'.")

if __name__ == "__main__":
    main()
