import sqlr.app as sqlr

# sql comments are labels for that particular code snippet
# the code snippets are returned like: snippet = sqlr.by_tag("tag_name"), returns a class wrapping the expression that fixtures can be applied to.
# the user adds fixtures: snippet.set_fixtures(dict[fixture_name, Fixture]); or snippet.set_fixture(fixture_name, Fixture)
# fixture name is the source name or alias used in the query.
# result: Fixture = snippet.run()  -- the output fixture can be used as input to another snippet
# sqlr.assert_equal(result, expected)

def test_plus2() -> None:
  testable_sql = sqlr.by_tag("plus_two")

  print(testable_sql)