with src as (
  select id, first_name, last_name from mytable
)
select
  *,
  upper(first_name) as first_name_upper,
  left(first_name, 1)
from src