SELECT c.*,
    f.template, f.row_disp_name, f.units, f.recorded_time_utc, f.num_value1, f.num_value2
FROM 
    `som-nero-phi-jonc101.triageTD.cohort_enc_code_age` as c -- old: traige_TE.triage_cohort_adjusted
JOIN 
    `som-nero-phi-jonc101.shc_core.flowsheet` as f
ON 
    (c.anon_id=f.anon_id and c.inpatient_data_id_coded=f.inpatient_data_id_coded)
WHERE
    recorded_time_utc < admit_time --, 'yyyy-mm-dd hh24:mi:ss'
AND
(
(row_disp_name="BP")
OR
(row_disp_name="NIBP")
OR
(row_disp_name="Resting BP")
OR
(row_disp_name="Pulse")
OR
(row_disp_name="Heart Rate")
OR
(row_disp_name="Resting HR")
OR
(row_disp_name="Weight")
OR
(row_disp_name="Height")
OR
(row_disp_name="Temp")
OR
(row_disp_name="Resp")
OR
(row_disp_name="Resp Rate")
OR
(row_disp_name="Resting RR")
OR
(row_disp_name="SpO2")
OR
(row_disp_name="Resting SpO2")
OR
(row_disp_name="O2 (LPM)")
OR
(row_disp_name="Arterial Systolic BP")
OR
(row_disp_name="Arterial Diastolic BP")
OR
(row_disp_name="Temp (in Celsius)")
OR
(row_disp_name="Blood Pressure")
OR
(row_disp_name="Oxygen Saturation")
OR
(row_disp_name="Glasgow Coma Scale Score")
OR
(row_disp_name="Altered Mental Status (GCS<15)")
OR
(row_disp_name="Total GCS Points")
OR
(row_disp_name="GCS Score")
OR
(row_disp_name= "Temp 2")
OR
(row_disp_name= "Temperature (Blood - PA line)")
OR
(row_disp_name= "Respiratory Rate")
OR
(row_disp_name= "O2 Flow (L/min)")
OR
(row_disp_name= "O2")
OR
(row_disp_name= "Activity")
OR
(row_disp_name= "Mobility")
OR
(row_disp_name= "acuity score")
OR
(row_disp_name= "Acuity as Level of Care")
OR
(row_disp_name= "LOC")
OR
(row_disp_name= "LOC Score")
)
