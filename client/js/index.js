function buildSelector(options) {
  var select = document.getElementById("answer"); 
  for(var i = 0; i < options.length; i++) {
      var opt = options[i];
      var el = document.createElement("option");
      el.textContent = opt.value;
      el.value = opt.id;
      select.appendChild(el);
  }
}

function loadData() {
  var xhttp = new XMLHttpRequest();
  xhttp.onreadystatechange = function() {
    if (this.readyState == 4 && this.status == 200) {
      response = JSON.parse(this.responseText)
      document.getElementById("question").innerHTML = response.question
      buildSelector(response.options)
    }
  };
  xhttp.open("GET", "http://localhost:10000/question", true);
  xhttp.send();
}
loadData();