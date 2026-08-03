select distinct
  department_id,
  upper(department_name) as department_name,
  cost_center,
  budget
from mydatabase.myschema.raw_department
