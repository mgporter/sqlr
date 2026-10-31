with 
plus_one as (
  -- this is plus one
  select num, num + 1 as num_plus_one
  from numbers
),
plus_two as (
  -- @sqlr(plus_two)
  select num, num + 2 as num_plus_two, name
  from numbers
),
plus_three as (
  -- this is plus three
  select num, num + 3 as num_plus_three
  -- @sqlr(incorrect_tag)
  from numbers
),
plus_four as (
  select num, num + 4 as num_plus_four
  from (  -- @sqlr(inner_select)
   select distinct num from numbers
  )
)
-- @sqlr(main)
select
  p1.num,
  p1.num_plus_one,
  p2.num_plus_two,
  p3.num_plus_three,
  p4.num_plus_four
from plus_one p1
left join plus_two p2 on p1.num = p2.num
left join plus_three p3 on p1.num = p3.num
left join plus_four p4 on p1.num = p4.num;