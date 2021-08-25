SELECT c.anon_id, c.pat_enc_csn_id_coded, c.inpatient_data_id_coded, c.admit_time,
    f.template, f.row_disp_name, f.units, f.recorded_time_utc, f.num_value1, f.num_value2
FROM 
    `som-nero-phi-jonc101.triageTD.1_2_cohort` as c 
JOIN 
    `som-nero-phi-jonc101.shc_core.flowsheet` as f
ON 
    (c.anon_id=f.anon_id and c.inpatient_data_id_coded=f.inpatient_data_id_coded)
WHERE
    recorded_time_utc < admit_time --, 'yyyy-mm-dd hh24:mi:ss'
AND row_disp_name in 
('Heart Rate', 'Pulse', "Resting HR", 'Resting Heart Rate (bpm)', 'Resting Pulse Rate: (Record BPM)', -- smaller number, might be too noisy
 'O2', 'O2 (LPM)', 'O2 Flow (L/min)', 'O2 Delivery Method', 
 'Resp Rate', 'Resp', 'Respiratory Rate', -- "Resting RR" not there
 'BP', 'NIBP', 'Arterial Systolic BP' , 'Arterial Diastolic BP' , 'Blood Pressure', "Resting BP", --'Resting Systolic Blood Pressure',
 'Temp', 'Temp (in Celsius)', 'Temperature (Blood - PA line)', 'Temp 2', 'Temperature', 
 'Activity', 'Mobility', 
 'acuity score', 'Acuity as Level of Care',
 'LOC', 'LOC Score')
-- removed GCS, too many missing and not consistent
--  'SpO2', "Resting SpO2", 'Oxygen Saturation', 'Resting O2 Saturation', -- difficult to interpret without O2 delivery