import networkx as nx

def are_neighbors(pos1, pos2):
    '''Return True if the two pixels are neighbors in the image.'''
    
    if abs(pos1[0]-pos2[0])<=1 and abs(pos1[1]-pos2[1]) <=1:
        return True
    else:
        return False
        
    
def in_region(pos, win_center, win_size):
    '''Returns True if the given position is inside the window.
    
    Parameters
    ----------
    pos : posição do ponto.
    win_center : centro da janela.       
    win_size : tamanho da janela.
    
    '''

    half_win_size = win_size // 2
    if pos[0] >= win_center[0] - half_win_size and pos[0] <= win_center[0] + half_win_size:
        if pos[1] >= win_center[1] - half_win_size and pos[1] <= win_center[1] + half_win_size:
            return True
    return False

def nodes_in_win(graph, win_center, win_size):
    '''Indica os nós que estão dentro da janela.
    
    Parameters
    ----------
    graph : o grapo a ser analisado.
    win_center : centro da janela.       
    win_size : tamanho da janela.
    
    '''
    pos_nodes = graph.nodes(data='center')
    #pos_nodes = pos_nodes[:,1:]
    nodes_inside =  []

    for item in pos_nodes:
        node = item[0]
        pos = item[1]
        if in_region(pos, win_center, win_size):
             nodes_inside.append(node)
               
    return nodes_inside

def translate_graph(graph, win_center, win_size):
    '''Transalada a grafo, muda os índices dos nós e das arestras.
    
    Parameters
    ----------
    graph : graph a ser transaladado. 
    win_center : centro da janela.       
    win_size : tamanho da janela.
    
    '''   
    
    r0 = win_center[0] - win_size//2
    c0 = win_center[1] - win_size//2
    new_pos_no = []
    pos_nodes = graph.nodes(data='center')

    for item in pos_nodes:
        node = item[0]
        pos = item[1]
        new_pos_no = [pos[0] - r0, pos[1] - c0]
        graph.nodes[node]['center'] = new_pos_no

    pos_edges = graph.edges(data='path', keys = True)
    for item in pos_edges:
        node1 = item[0]
        node2 = item[1]
        key = item[2]
        new_pos_edge = []
        pos_edge = item[3]
        for pos in pos_edge:
            new_pos_edge_pixel = [pos[0] - r0, pos[1] - c0]
            new_pos_edge.append(new_pos_edge_pixel)
        graph[node1][node2][key]['path'] = new_pos_edge
    
    return graph

def edges_in_win(graph, win_center, win_size):
    '''Define a arestras que estão dentro da janela, incluindo as que estão cruzando e os nós que
    as conectam, mesmo que ambos estejam fora da janela.
    
    Parameters
    ----------
    graph : o grafo a ser analisado.
    win_center : centro da janela.       
    win_size : tamanho da janela.
    
    '''
        
    pos_edges = graph.edges(data='path', keys = True)
    #pos_nodes = pos_nodes[:,1:]
    edges_inside =  []
    
    edges_inside = set()

    for item in pos_edges:
        node1 = item[0]
        node2 = item[1]
        key = item [2]
        pos_edge = item[3]
        for pos in pos_edge:
            edge = (node1,  node2, key)
            if in_region(pos, win_center, win_size) and (edge not in edges_inside):
                 edges_inside.add(edge)
               
    return edges_inside

def follow_edge(pos_edge_aug, win_center, win_size):
    '''For a given edge, identifies the positions where the edge crosses the window. The edge is broken
    at those positions and new edges are created for each part of the edge inside the window.
    
    Parameters
    ----------
    pos_edge_aug : List
        List of pixels
    win_center : tuple of int
        The center of the window
    win_size : int
        The size of the window
        
    Returns
    -------
    new_edges_pos : List
        Each item of this list contains the pixels of a new edge inside the window. These edges are
        portions of the original edge (`pos_edge_aug`) that are inside the window. Note that the first
        and last elements of a given edge are actually the position of the nodes that should connect
        to the edge.
    '''
    
    curr_inside = in_region(pos_edge_aug[0], win_center, win_size)
    new_edges_pos = []  # Stores the new edges generated from a single edge crossing the window
    new_edge_pos = []   # Stores the position of a new edge
    for idx in range(len(pos_edge_aug)-1):
        pos = pos_edge_aug[idx]
        next_inside = in_region(pos_edge_aug[idx+1], win_center, win_size)
        if curr_inside==True:
            # If we are inside the window, add current position to the new edge
            new_edge_pos.append(pos)
            if next_inside==False:
                # If the next position is outside the edge, store the current edge
                # and start a new one.
                new_edges_pos.append(new_edge_pos)
                new_edge_pos = []
                
        # Set `curr_inside` for the next iteration of the loop
        curr_inside = next_inside
        
    if curr_inside==True:
        # Add the last point if it is inside the window
        idx += 1
        pos = pos_edge_aug[idx]
        new_edge_pos.append(pos)
        new_edges_pos.append(new_edge_pos)
        
    return new_edges_pos

def add_edges_to_graph(graph, new_edges_pos):
    '''Given a list of edges positions `new_edges_pos`, add the edges to `graph`.
    
    `new_edges_pos` contains the positions of the edges to be added. In addition, the first and last
    elements of each edge in `new_edges_pos` contains the positions of the nodes connected by the edge.
    
    
    Parameters
    ----------
    graph : nx.MultiGraph
        The graph 
    new_edges_pos : List
        Positions of the edges as well as the respective nodes (see function description).
        
    Returns
    -------
    None
    '''
    
    curr_node = len(graph)
    
    nodes_pos_dict = {pos:node for node, pos in graph.nodes(data='center')}
    for new_edge_pos in new_edges_pos:
        if len(new_edge_pos)>2:   # !! Edges smaller than 2 pixels are removed from the image !!
            node1_pos = new_edge_pos[0]
            node2_pos = new_edge_pos[-1]
            if node1_pos in nodes_pos_dict:
                node1 = nodes_pos_dict[node1_pos]
            else:
                node1 = curr_node
                nodes_pos_dict[node1_pos] = curr_node
                curr_node += 1
                graph.add_node(node1, center=node1_pos)
            if node2_pos in nodes_pos_dict:
                node2 = nodes_pos_dict[node2_pos]
            else:
                node2 = curr_node
                nodes_pos_dict[node2_pos] = curr_node
                curr_node += 1
                graph.add_node(node2, center=node2_pos)

            graph.add_edge(node1, node2, path=new_edge_pos[1:-1])
            
def graph_in_window(graph, win_center, win_size):
    '''For each edge crossing the window, remove the portion of the edge outside of the window, create 
    a node at the crossing point (window border) and connect the newly created node with the node inside 
    the window that is connected to the edge. 
    
    Note that the same edge might cross the window many times. Also, the two nodes connected to an edge 
    might be both outside of the window. 
    
    Warning, strictly speaking, the above mentioned operation may create isolated nodes (with degree 0) 
    and edges with size 0 (i.e., nodes that are neighbors in the image). The function removes such cases,
    which means that the returned graph is not exactly the same as the original graph inside the window.
    
    Parameters
    ----------
    graph : nx.MultiGraph
        The original graph
    win_center : tuple of int
        The center of the window
    win_size : int
        The size of the window
        
    Returns
    -------
    subgraph_trans : nx.MultiGraph
        The graph inside the window. The positions of the nodes and edges are respective to the window,
        not the original image.
    '''

    edges_inside = edges_in_win(graph, win_center, win_size)

    subgraph = nx.MultiGraph()
    for node1, node2, key in edges_inside:

        pos_node1 = graph.nodes[node1]['center']
        pos_node2 = graph.nodes[node2]['center']
        pos_edge = graph[node1][node2][key]['path']

        # Add nodes positions to the edge path. This simplifies the analysis.
        pos_edge_aug = pos_edge
        if are_neighbors(pos_edge[0], pos_node1):
            pos_edge_aug = [pos_node1] + pos_edge_aug + [pos_node2]
        elif are_neighbors(pos_edge[0], pos_node2):
            pos_edge_aug = [pos_node2] + pos_edge_aug + [pos_node1]
        else:
            print('Warning! Edge is not a neighbor of either node.')

        new_edges_pos = follow_edge(pos_edge_aug, win_center, win_size)
        add_edges_to_graph(subgraph, new_edges_pos)

    subgraph_trans = translate_graph(subgraph, win_center, win_size)
    
    return subgraph_trans