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
    Question    string      `json:"question"`    
    Options     []Option    `json:"options"`

}

var Response QuestionResponse

func enableCors(w *http.ResponseWriter) {
    (*w).Header().Set("Access-Control-Allow-Origin", "*")
}

func homePage(w http.ResponseWriter, r *http.Request) {
    fmt.Fprintf(w, "Welcome to the HomePage!")
    fmt.Println("Endpoint Hit: homePage")
}

func returnQuestion(w http.ResponseWriter, r *http.Request) {
    enableCors(&w)
    fmt.Println("Endpoint Hit: Question")
    json.NewEncoder(w).Encode(Response)
}

func returnRecommend(w http.ResponseWriter, r *http.Request) {
    enableCors(&w)
    fmt.Println("Endpoint Hit: Recommend")
    cmd := exec.Command("python3", "recommender/server.py")
    cmd.Stdout = os.Stdout
    cmd.Stderr = os.Stderr
    log.Println(cmd.Run())
    json.NewEncoder(w).Encode(Response)
}


func handleRequests() {
    myRouter := mux.NewRouter().StrictSlash(true)
    myRouter.HandleFunc("/", homePage)
    myRouter.HandleFunc("/question", returnQuestion)
    myRouter.HandleFunc("/recommend", returnRecommend)
    log.Fatal(http.ListenAndServe(":10000", myRouter))
}

func main() {
    Response = QuestionResponse{Question: "How are you feeling?", Options: []Option{
        Option{Id: "1", Value:"😊Happy😊"},
        Option{Id: "2", Value:"😔Sad😔"},
    }}
    handleRequests()
    log.Println("Running")
}