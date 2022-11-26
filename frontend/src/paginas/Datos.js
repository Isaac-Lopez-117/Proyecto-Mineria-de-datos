import React, { Component } from "react";
import { Col, Container, Row } from "reactstrap";
import ProjectList from "../components/ProjectList";
import NewProjectModal from "../components/NewProjectModal";

import axios from "axios";

import { API_URL } from "../constants";

class Datos extends Component {
  state = {
    projects: [],
  };

  componentDidMount() {
    this.resetState();
  }

  getProjects = () => {
    axios.get(API_URL).then((res) => this.setState({ projects: res.data }));
  };

  resetState = () => {
    this.getProjects();
  };

  render() {
    return (
      <Container style={{ marginTop: "20px" }}>
        <Row>
          <Col>
            <ProjectList
              projects={this.state.projects}
              resetState={this.resetState}
            />
          </Col>
        </Row>
        <Row>
          <Col>
            <NewProjectModal create={true} resetState={this.resetState} />
          </Col>
        </Row>
      </Container>
    );
  }
}

export default Datos;
