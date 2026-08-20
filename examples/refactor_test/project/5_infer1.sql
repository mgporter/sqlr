select
  upper(name) as name_upper,
  cast(x as integer) * 2 as double_x
from undeclared_table