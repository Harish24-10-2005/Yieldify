document.addEventListener("DOMContentLoaded", function() {
    const apiURL = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070?api-key=579b464db66ec23bdd000001b4ce6b88e26246144075f725f7ee1ba9&format=json&limit=20000";
    let data = [];
  
    fetch(apiURL)
        .then(response => response.json())
        .then(jsonData => {
            data = jsonData.records;
            populateFilters(data);
            displayData(data);
        })
        .catch(error => console.error('Error fetching data:', error));
  
    function populateFilters(data) {
        const stateSet = new Set();
        const districtSet = new Set();
        const commoditySet = new Set();
  
        data.forEach(item => {
            stateSet.add(item.state);
            districtSet.add(item.district);
            commoditySet.add(item.commodity);
        });
  
        populateSelect(document.getElementById('state'), stateSet);
        populateSelect(document.getElementById('district'), districtSet);
        populateSelect(document.getElementById('commodity'), commoditySet);
    }
  
    function populateSelect(selectElement, dataSet) {
        dataSet.forEach(item => {
            const option = document.createElement('option');
            option.value = option.textContent = item;
            selectElement.appendChild(option);
        });
    }
  
    function displayData(filteredData) {
        const tableBody = document.getElementById('data-table').querySelector('tbody');
        tableBody.innerHTML = '';
  
        filteredData.forEach(item => {
            const row = document.createElement('tr');
            Object.keys(item).forEach(key => {
                const cell = document.createElement('td');
  
                // Check if the key is one of the price keys and divide by 100
                if (key === 'min_price' || key === 'max_price' || key === 'modal_price') {
                    cell.textContent = (item[key] / 100).toFixed(2);
                } else {
                    cell.textContent = item[key];
                }
  
                row.appendChild(cell);
            });
            tableBody.appendChild(row);
        });
    }
  
    window.filterData = function() {
        const state = document.getElementById('state').value;
        const district = document.getElementById('district').value;
        const commodity = document.getElementById('commodity').value;
  
        const filteredData = data.filter(item => {
            return (!state || item.state === state) &&
                   (!district || item.district === district) &&
                   (!commodity || item.commodity === commodity);
        });
  
        displayData(filteredData);
    }
  
  });
  