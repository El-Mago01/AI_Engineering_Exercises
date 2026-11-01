This is a bigger learning project with quite some nice learnings:
- Using a TypeDict Class. Turning a class into a simple dictionary
- Using total=false in the class definition: class TravelState(TypeDict, total=False):
- Using keys that are annotated types, this ensures the adding, not overwritting of a finding. In this case an operator.add. This is called a reducer. See findings: Annotated[list[dict], operator.add]
- 