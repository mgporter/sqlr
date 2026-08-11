select
  id,
  name,
  date_trunc('day', modified_at) as modified_at,
  list_extract(titles, 1) as first_title
from mydatabase.test
where modified_at > cast('2023-01-01 00:00:00' as timestamp_ntz(6))
