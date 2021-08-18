select distinct lab.order_proc_id_coded,
                lab.order_type,
                lab.proc_code as code,
                lab.order_time_jittered_utc as lab_order,
                labels.*
from
  conor_db.er_empiric_treatment as labels,
  starr_datalake2018.order_proc as lab
where
  lab.order_type in ('Lab', 'Microbiology', 'Microbiology Culture')
  and labels.jc_uid = lab.jc_uid
  and timestamp_add(lab.order_time_jittered_utc, INTERVAL 1 HOUR) < labels.admit_time
  and timestamp_add(lab.order_time_jittered_utc, INTERVAL 24*365 HOUR) >= labels.admit_time
