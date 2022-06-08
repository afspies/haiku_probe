AUTHORS="Alex Spies"
YEAR="2022"
PROJECT_NAME="haiku_template"

mv haiku_template/tests/test_haiku_template.py haiku_template/tests/test_$PROJECT_NAME.py
mv haiku_template/haiku_template.py haiku_template/$PROJECT_NAME.py
mv haiku_template $PROJECT_NAME

find ./ -type f -not -path "./setup_template.sh" -exec sed -i -e 's/haiku_template/'"$PROJECT_NAME"'/g' {} \;
find ./ -type f -not -path "./setup_template.sh" -exec sed -i -e 's/\[\[AUTHORS\]\]/"'"$AUTHORS"'"/g' {} \;
find ./ -type f -not -path "./setup_template.sh" -exec sed -i -e 's/\[\[YEAR\]\]/'"$YEAR"'/g' {} \;