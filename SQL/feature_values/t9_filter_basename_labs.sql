# output: traige_cohort_2019_labs_basename_filtered
select * 
from traige_TE.triage_cohort_2019_all_labs as labs
where labs.base_name in ('AG','AGAP','ALB','ALT','AST','BASOAB','BE','BUN','CA','CL','CO2','CR','EOSAB','GLOB','GLU','HCO3','HCO3A','HCO3V','HCT','HGB','INR','K','LAC','LACWBL','LYMAB','MG','MONOAB','NEUTAB','O2SATA','O2SATV','PCAGP','PCBUN','PCCL','PCO2A','PCO2V','PH','PHA','PHCAI','PHOS','PHV','PLT','PO2A','PO2V','PT','RBC','TBIL','TCO2A','TCO2V','TNI','UPH','WBC','XLEUKEST','XUKET')
and labs.pat_enc_csn_id_coded in
(select distinct(cohort.pat_enc_csn_id_coded) from traige_TE.triage_cohort_final as cohort)