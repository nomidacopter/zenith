// main.go
package main

import (
    "encoding/json"
    "fmt"
    "log"
    "net/http"

    "os"
    "os/exec"

    "github.com/gorilla/mux"
)

type Option struct {
    Id      string  `json:"id"`
    Value   string  `json:"value"`
}

type QuestionResponse struct {
    Question    string              `json:"question"`    
    Options     []Option            `json:"options"`
    nextRoute   string              `json:"nextRoute"`
    params      map[string]string   `json:"params"`

}

var Response QuestionResponse

func enableCors(w *http.ResponseWriter) {
    (*w).Header().Set("Access-Control-Allow-Origin", "*")
}

func getQueryAnswer(r *http.Request) string {
    query := r.URL.Query()
    answer, present := query["answer"]
    if !present || len(answer) == 0 {
        fmt.Println("answer is not present")
    }
    return answer[0]
}

func homePage(w http.ResponseWriter, r *http.Request) {
    fmt.Fprintf(w, "Welcome to the HomePage!")
    fmt.Println("Endpoint Hit: homePage")
}

func returnFeelings(w http.ResponseWriter, r *http.Request) {
    enableCors(&w)
    fmt.Println("Endpoint Hit: Feelings")
    question := "How are you feeling?"
    feelings := GetFeelings()
    nextRoute := "/recommend"
    Response = QuestionResponse{Question: question, Options: feelings, nextRoute: nextRoute}
    json.NewEncoder(w).Encode(Response)
}

func returnRecommend(w http.ResponseWriter, r *http.Request) {
    enableCors(&w)
    fmt.Println("Endpoint Hit: Recommend")
    answer := getQueryAnswer(r)
    fmt.Println(answer)
    cmd := exec.Command("python3", "recommender/server.py")
    cmd.Stdout = os.Stdout
    cmd.Stderr = os.Stderr
    log.Println(cmd.Run())
    json.NewEncoder(w).Encode(Response)
}

func returnWhen(w http.ResponseWriter, r *http.Request) {
    enableCors(&w)
    fmt.Println("Endpoint Hit: When")
    question := "When do you want to do it?"
    feelings := GetFeelings()
    nextRoute := "/feelings"
    Response = QuestionResponse{Question: question, Options: feelings, nextRoute: nextRoute}
    json.NewEncoder(w).Encode(Response)
}


func processAnswer(w http.ResponseWriter, r *http.Request) {
    enableCors(&w)
    fmt.Println("Endpoint Hit: Answer")
    AppendText("wowzers")
    json.NewEncoder(w).Encode(Response)
}

func handleRequests() {
    myRouter := mux.NewRouter().StrictSlash(true)
    myRouter.HandleFunc("/", homePage)
    myRouter.HandleFunc("/feelings", returnFeelings)
    myRouter.HandleFunc("/recommend", returnRecommend)
    myRouter.HandleFunc("/feelings", returnWhen)

    myRouter.HandleFunc("/answer", processAnswer)
    log.Fatal(http.ListenAndServe(":10000", myRouter))
}

func main() {
    log.Println("Running")

    handleRequests()
}