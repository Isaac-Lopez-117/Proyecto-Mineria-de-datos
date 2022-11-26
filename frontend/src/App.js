import React, { Component, Fragment } from "react";
import { BrowserRouter as Router, Switch, Route } from "react-router-dom";
import Navbar from "./components/NavBar";
import About from "./paginas/About";
import Home from "./paginas/Home";
import Datos from "./paginas/Datos";
import EDA from "./paginas/EDA";
import PCA from "./paginas/PCA";

class App extends Component {
  render() {
    return (
      <Router>
        <Navbar />
        <div className="container p-2">
          <Switch>
            <Route path="/datos">
              <Datos />
            </Route>
            <Route path="/eda">
              <EDA />
            </Route>
            <Route path="/pca">
              <PCA />
            </Route>
            <Route path="/about">
              <About />
            </Route>
            <Route path="/">
              <Home />
            </Route>
          </Switch>
        </div>
      </Router>
    );
  }
}

export default App;
