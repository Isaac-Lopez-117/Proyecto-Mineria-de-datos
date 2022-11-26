import React, { Component } from "react";
import { Table } from "reactstrap";
import NewProjectModal from "./NewProjectModal";

import ConfirmRemovalModal from "./ConfirmRemovalModal";

class ProjectList extends Component {
  render() {
    const projects = this.props.projects;
    return (
      <Table dark>
        <thead>
          <tr>
            <th>Name</th>
            <th>URL</th>
            <th>Description</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          {!projects || projects.length <= 0 ? (
            <tr>
              <td colSpan="6" align="center">
                <b>Ops, no one here yet</b>
              </td>
            </tr>
          ) : (
            projects.map((project) => (
              <tr key={project.pk}>
                <td>{project.name}</td>
                <td>{project.url}</td>
                <td>{project.desc}</td>
                <td align="center">
                  <NewProjectModal
                    create={false}
                    project={project}
                    resetState={this.props.resetState}
                  />
                  &nbsp;&nbsp;
                  <ConfirmRemovalModal
                    pk={project.pk}
                    resetState={this.props.resetState}
                  />
                </td>
              </tr>
            ))
          )}
        </tbody>
      </Table>
    );
  }
}

export default ProjectList;
