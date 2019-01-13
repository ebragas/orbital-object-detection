URL="https://us-central1-reliable-realm-222318.cloudfunctions.net/hello_world"

curl -X POST ${URL} -H "Content-Type:application/json" -d '{"name": "Eric"}'

curl ${URL}
