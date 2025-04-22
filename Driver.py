# test_semantic_router.py
from semantic_router import Route, SemanticRouter
from semantic_router.encoders import OpenAIEncoder
import os
from dotenv import load_dotenv
import time
import json
# Load environment variables from .env file
load_dotenv()

print("Successfully imported Semantic Router!")

# Get the OpenAI API key from environment variables
openai_api_key = os.environ.get("OPENAI_API_KEY")

if not openai_api_key:
    print("Warning: OPENAI_API_KEY not found in environment variables.")
else:
    try:
        # Initialize the encoder with the correct parameter name
        encoder = OpenAIEncoder(openai_api_key=openai_api_key)

        router = SemanticRouter(encoder=encoder)
        
        # Add routes explicitly
        print("Adding routes to the router...")

        with open('saved_routes (Reference).json', 'r', encoding='utf-8') as file:
            data = json.load(file)
        all_routes = []
        
        for route_data in data:
            route = Route(name=route_data['name'], utterances=route_data['utterances'])
            all_routes.append(route)
        
        router.add(all_routes)

        print('Number of routes added to the router: ', len(all_routes))
        
        # Wait a moment to ensure the index is ready
        print("Waiting for index to be ready...")
        time.sleep(3)
        
        
        results = []
        test_queries = [
                        "Users in the specified RAN region should experience at least 30 Mbps average download speed" ,
                        "Ensure users in the designated RAN region receive an average download speed of no less than 30 Mbps" ,
                        "Users located in the specified RAN area must have an average download speed of 30 Mbps or above",
                        "People in that RAN zone need to get around 30 Mbps download speed, minimum",
                        "Keep bad SINR below 10% of the cells in this RAN zone.",
                        "The operator should ensure that fewer than 10% of the cells suffer from low SINR in this RAN region.",
                        "Low SINR should not affect more than 10% of the cells in this RAN region.",
                        "In the rural macrocell area, the RAN infrastructure should operate at more than 80% energy efficiency",
                        "The RAN infrastructure in rural macrocells should function with energy efficiency above 80%.",
                        "The number of simultaneously active users in the test RAN area must be limited to 200 or fewer.",
                        "If more than 200 users are active simultaneously in the test RAN area, system performance may degrade.",
                        "Ensure the number of PDU sessions in Urban Core is greater than 100000",
                        "Confirm that the Urban Core maintains more than 100000 PDU sessions.",
                        "The PDU session count in the Urban Core should be greater than 100000.",
                        "Confirm that the throughput for External DN traffic reaches no less than 10 Gbps.",
                        "External DN traffic must achieve a throughput of no less than 10 Gbps.",
                        "Make sure the External DN line runs at 10 Gbps or more."
                        ]
        # Test the router
        for i in range(len(test_queries)):
            test_query = test_queries[i]
            result = router(test_query)
            results.append(result.name)
        print(f"The final results are:.\n {results}")
        
    except Exception as e:
        print(f"Error when testing router: {e}")
        import traceback
        traceback.print_exc()

print("Setup complete!")