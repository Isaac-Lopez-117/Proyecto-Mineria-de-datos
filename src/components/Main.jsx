import React from "react";
import { BrowserRouter as Router, Switch, Route } from "react-router-dom";
// Barra de navegacion
import Navbar from "./Navbar";
// Paginas
import EDA from "../paginas/EDA";
import PCA from "../paginas/PCA";
import Home from "../paginas/Home";

/* NOTA IMPORTANTE: Para utilizar correctamente router utilizar el comando
npm install react-router-dom@5.2.0
para instalar su version funcional */

const Main = () => {
  return (
    <Router>
      <div>
        <Navbar />
        <hr />
        <Switch>
          <Route path="/home">
            <Home />
          </Route>
          <Route path="/EDA">
            <EDA />
          </Route>
          <Route path="/PCA">
            <PCA />
          </Route>
          <Route path="/">Este es la raiz</Route>
        </Switch>
      </div>
    </Router>
  );
};

export default Main;
