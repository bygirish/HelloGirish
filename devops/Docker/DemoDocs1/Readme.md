





```
Service names and corresponding docker hub images to be used in yaml files:

vote : ritikbilala/dockervotingapp_vote:latest
result : ritikbilala/dockervotingapp_result:latest
db : 
   image: postgres:9.4 
   environment:
       POSTGRES_USER: "postgres"
       POSTGRES_PASSWORD: "postgres"
worker : ritikbilala/dockervotingapp_worker:latest
redis : redis:alpine

```