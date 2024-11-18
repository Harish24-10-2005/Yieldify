// Import the axios library
const axios = require('axios');

// Define the options for the request
const options = {
  method: 'POST',
  url: 'https://microsoft-translator-text.p.rapidapi.com/BreakSentence',
  params: {
    'api-version': '3.0'
  },
  headers: {
    'content-type': 'application/json',
    'X-RapidAPI-Key': 'c89947fe7cmsh03552616a4dcfd6p1ccfc5jsn3239d9e638c1',
    'X-RapidAPI-Host': 'microsoft-translator-text.p.rapidapi.com'
  },
  data: [
    {
      Text: 'How are you? I am fine. What did you do today?'
    }
  ]
};

// Define an async function to make the request
async function makeRequest() {
  try {
    // Await the response from the axios request
    const response = await axios.request(options);
    // Log the response data to the console
    console.log(response.data);
  } catch (error) {
    // Log any errors to the console
    console.error(error);
  }
}

// Call the async function to execute the request
makeRequest();
