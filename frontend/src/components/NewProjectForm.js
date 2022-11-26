import React from "react";
import { Button, Form, FormGroup, Input, Label } from "reactstrap";

import axios from "axios";

import { API_URL } from "../constants";

class NewProjectForm extends React.Component {
  state = {
    pk: 0,
    name: "",
    url: "",
    desc: "",
  };

  componentDidMount() {
    if (this.props.project) {
      const { pk, name, url, desc } = this.props.project;
      this.setState({ pk, name, url, desc });
    }
  }

  onChange = (e) => {
    this.setState({ [e.target.name]: e.target.value });
  };

  createProject = (e) => {
    e.preventDefault();
    axios.post(API_URL, this.state).then(() => {
      this.props.resetState();
      this.props.toggle();
    });
  };

  editProject = (e) => {
    e.preventDefault();
    axios.put(API_URL + this.state.pk, this.state).then(() => {
      this.props.resetState();
      this.props.toggle();
    });
  };

  defaultIfEmpty = (value) => {
    return value === "" ? "" : value;
  };

  render() {
    return (
      <Form
        onSubmit={this.props.project ? this.editProject : this.createProject}
      >
        <FormGroup>
          <Label for="name">Name:</Label>
          <Input
            type="text"
            name="name"
            onChange={this.onChange}
            value={this.defaultIfEmpty(this.state.name)}
          />
        </FormGroup>
        <FormGroup>
          <Label for="url">URL:</Label>
          <Input
            type="text"
            name="url"
            onChange={this.onChange}
            value={this.defaultIfEmpty(this.state.url)}
          />
        </FormGroup>
        <FormGroup>
          <Label for="desc">Description:</Label>
          <Input
            type="text"
            name="desc"
            onChange={this.onChange}
            value={this.defaultIfEmpty(this.state.desc)}
          />
        </FormGroup>
        <Button>Send</Button>
      </Form>
    );
  }
}

export default NewProjectForm;
