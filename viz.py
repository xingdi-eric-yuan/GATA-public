from typing import Iterable, Tuple

import numpy as np
import networkx as nx
import plotly
import plotly.graph_objects as go
import matplotlib.pylab as plt

from textworld.logic import Proposition

import generic



def build_graph_from_facts(facts: Iterable[Proposition]) -> nx.DiGraph:
    triplets = []
    for fact in facts:
        if fact.name in generic.ignore_relations:
            continue

        triplet = (*fact.names, fact.name)
        triplets.append(triplet)

    return build_graph_from_triplets(triplets)


def build_graph_from_triplets(triplets: Iterable[Tuple]) -> nx.DiGraph:
    G = nx.DiGraph()
    labels = {}
    for triplet in triplets:
        triplet = tuple(triplet)
        triplet = triplet if len(triplet) >= 3 else triplet + ("is",)

        src = triplet[0]
        dest = triplet[1]
        relation = triplet[-1]
        if relation in {"is"}:
            dest = src + "-" + dest

        if dest in ["somewhere", "open", "closed"]:
            dest = src + "-" + dest

        if src in ["exit"]:
            src = tuple(sorted([src, dest, triplet[2]]))

        labels[src] = triplet[0]
        labels[dest] = triplet[1]
        G.add_edge(src, dest, type=triplet[-1])

    nx.set_node_attributes(G, labels, 'label')
    return G


def show_kg(facts: Iterable[Proposition], title="Knowledge Graph", renderer=None, save=None):
    facts = list(facts)
    if facts and isinstance(facts[0], Proposition):
        G = build_graph_from_facts(facts)
    else:
        G = build_graph_from_triplets(facts)

    plt.figure(figsize=(16, 9))
    #pos = nx.layout.spring_layout(G, k=10)
    pos = nx.drawing.nx_agraph.graphviz_layout(G, prog="sfdp")

    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='text',
        mode='lines')

    trace3_list = []
    middle_node_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode='text',
        hoverinfo='none',

        marker=go.scatter.Marker(
            opacity=0,
        )
    )

    def _get_angle2(p0, p1):
        x0, y0 = p0
        x1, y1 = p1
        return np.rad2deg(np.arctan2((y1-y0), (x1-x0)))

    tmp = {}
    for edge in G.edges(data=True):
        trace3 = go.Scatter(
            x=[],
            y=[],
            mode='lines',
            line=dict(width=0.5, color='#888', shape='spline', smoothing=1),
            hoverinfo='none'
        )
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        rvec = (x0-x1, y0-y1)  # Vector from dest -> src.
        length = np.sqrt(rvec[0] ** 2 + rvec[1] ** 2)
        mid = ((x0+x1)/2., (y0+y1)/2.)
        orthogonal = (rvec[1] / length, -rvec[0] / length)

        trace3['x'] += (x0, mid[0] + 0 * orthogonal[0], x1, None)
        trace3['y'] += (y0, mid[1] + 0 * orthogonal[1], y1, None)
        trace3_list.append(trace3)

        offset_ = 5
        tmp[(pos[edge[0]], pos[edge[1]])] = (mid[0] + offset_ * orthogonal[0],
                                             mid[1] + offset_ * orthogonal[1])

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='text',
        text=["<b>{}</b>".format(data['label'].replace(" ", "<br>")) for n, data in G.nodes(data=True)],
        textfont=dict(
            family="sans serif",
            size=12,
            color="black"
        ),
        hoverinfo='none',
        marker=dict(
            showscale=True,
            # colorscale options
            #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
            #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
            #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
            color=[],
            size=10,
            line_width=2))

    fig = go.Figure(data=[*trace3_list, node_trace],
             layout=go.Layout(
                title=title,
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )

    def _get_angle(p0, p1):
        x0, y0 = p0
        x1, y1 = p1
        if x1 == x0:
            return 0

        angle = -np.rad2deg(np.arctan((y1-y0)/(x1-x0)/(16/9)))
        return angle


    fig.update_layout(
        annotations=[
            go.layout.Annotation(
                x=pos[edge[1]][0],
                y=pos[edge[1]][1],
                ax=(pos[edge[0]][0]+pos[edge[1]][0])/2,
                ay=(pos[edge[0]][1]+pos[edge[1]][1])/2,
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=3,
                arrowwidth=0.5,
                arrowcolor="#888",
                standoff=5 + np.log(90 / abs(_get_angle(pos[edge[0]], pos[edge[1]]))) * max(map(len, G.nodes[edge[1]]['label'].split())),
            )
        for edge in G.edges(data=True)] + [
            go.layout.Annotation(
                x=tmp[(pos[edge[0]], pos[edge[1]])][0],
                y=tmp[(pos[edge[0]], pos[edge[1]])][1],
                showarrow=False,
                text="<i>{}</i>".format(edge[2]['type']),
                textangle=_get_angle(pos[edge[0]], pos[edge[1]]),
                font=dict(
                    family="sans serif",
                    size=12,
                    color="blue"
                ),
            )
        for edge in G.edges(data=True)]
    )

    if renderer:
        fig.show(renderer=renderer)

    if save:
        #plotly.offline.plot(fig, filename = '/tmp/filename.html', auto_open=False)
        fig.write_image(save, width=1920, height=1080, scale=4)

    return fig
